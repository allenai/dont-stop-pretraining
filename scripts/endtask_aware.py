import torch
from collections import defaultdict, namedtuple
from torch.nn.utils import clip_grad_norm_
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.modules import FeedForward
from allennlp.data import Vocabulary
import numpy as np
from transformers import (
	AutoModelWithLMHead,
	AdamW,
)


import nvidia_smi
import sys
PATH="/home/ldery/internship/dsp/dont_stop_pretraining"
sys.path.insert(1, PATH)
from models import BasicClassifierWithF1
from data.dataset_readers.text_classification_json_reader_with_sampling import TextClassificationJsonReaderWithSampling
from modules.seq2vec_encoders.cls_pooler import CLSPooler
PATH="/home/ldery/internship/meta4da/algorithms"
sys.path.insert(1, PATH)
from utils import *
from .alpha_generator import *
import pdb
from tqdm import tqdm
import torch.nn.init as init
import math

# Setting up gpu usage monitoring
nvidia_smi.nvmlInit()
gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())


def print_gpu_stats():
	mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
	print('Gpu mem: {} (GiB)'.format(mem_res.used / (1024**2)))  # usage in GiB
	print('Gpu used: {} %'.format(100 * (mem_res.used / mem_res.total)))

# Calculates the norm of a list of vectors
def calc_norm(grads):
	norm = 0.0
	for g_ in grads:
		if g_ is not None:
			norm += (g_**2).sum()
	return np.sqrt(norm.item())

# Calculates the dot products of 2 gradient vectors
def dot_prod(g1, g2):
	total = 0.0
	for p1, p2 in zip(g1, g2):
		if p1 is None or p2 is None:
			continue
		total += (p1 * p2).sum()
	total = total.item() if isinstance(total, torch.Tensor) else total
	return total

# Get the starting index of the heads of the network
def get_body_end(model):
	pos = 0
	for k, _ in model.named_parameters():
		if '_text_field_embedder' in k:
			pos += 1
	return pos


class ModelWithAuxTasks(AutoModelWithLMHead):
	''' Wrapper around a basic model so we can do gradient surgery on the model'''
	def __init__(
					self,
					model_name,
					base_lm_model,
					base_task_dataset_files,  # Dictionary of task_id : task_file
					step_frequency=1,
					max_seq_len=512,
					dropout=0.0,
					embedding_dim=768,
					ff_multiplier=1,
					num_layers=1,
					max_norm=1.0,
					save_path=None,
					alpha_generator_algo='default',
					primary_task_id=None,
					batch_sz=8,
					prim_dev_file =None, 
					prim_test_file =None,
					grad_accum_factor=8,
					no_mlm_weight=False
	):
		assert save_path is not None, 'Invalid Save Path Provided for Classifier Head'
		assert isinstance(base_task_dataset_files, dict), 'Invalid type of base_task_dataset_files. Expected array'
		assert primary_task_id is not None, 'No primary task id is given'
		assert primary_task_id in base_task_dataset_files, 'Primary Task not included in the list of dataset files'
		self.save_path = save_path
		self.base_lm_model = base_lm_model
		self.base_task_dataset_files = base_task_dataset_files
		self.datasets = self.setup_datasets(base_task_dataset_files, model_name, max_seq_len, lazy=False)
		# Cached for later use
		self.embedding_dim = embedding_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.ff_multiplier = ff_multiplier
		self.setup_classifiers(dropout, embedding_dim, num_layers, ff_multiplier)
		self.batch_sz = batch_sz
		self.max_norm = 1.0
		self.max_seq_len = max_seq_len
		self.model_name = model_name
		self.primary_task_id = primary_task_id
		self.prim_dev_file = prim_dev_file
		self.prim_test_file = prim_test_file
		self.prim_train_dataset = None
		self.prim_dev_dataset = None
		self.prim_test_dataset = None
		self.grad_accum_factor = grad_accum_factor
		self.no_mlm_weight = no_mlm_weight
		self.MLM_grads = None


	# Sets up the weighting generator
	def setup_alpha_generator(self, options):
		# Create a special task called MLM.
		task_names = list(self.base_task_dataset_files.keys())
		task_names.append("MLM")
		# Remove the primary task name
		aux_tasks = [x for x in task_names if x != self.primary_task_id]
		self.aux_tasks = aux_tasks
		self.alpha_generator_algo = get_alpha_generator(options, self.primary_task_id, aux_tasks)
		# Setup datastructures for logging
		if self.alpha_generator_algo.is_meta:
			self.options = options
			self.dp_stats = defaultdict(list)
			self.weight_stats = defaultdict(list)


	def setup_datasets(self, base_task_dataset_files, model_name, max_seq_len, label_vocab=None, lazy=False):
		# Instantiate dataset reader
		datasets = defaultdict(dict)
		indexers = {'tokens': PretrainedTransformerIndexer(model_name, do_lowercase=False)}
		tokenizer = PretrainedTransformerTokenizer(model_name, do_lowercase=False, start_tokens=["<s>"], end_tokens=["</s>"])
		dataset_reader = TextClassificationJsonReaderWithSampling(
							token_indexers=indexers, tokenizer=tokenizer,
							max_sequence_length=max_seq_len, lazy=lazy
						)
		# Read from the dataset
		pretrain_vocab = tokenizer._tokenizer.encoder
		for idx_, fname in base_task_dataset_files.items():
			all_samples = dataset_reader._read(fname)
			all_sentences, all_instances = [], []
			for instance in all_samples:
				tokens = instance.fields['tokens']
				tokens.index(pretrain_vocab)
				sentence = tokens.as_tensor(tokens.get_padding_lengths())['tokens']
				all_sentences.append(sentence)
				all_instances.append(instance)
			if label_vocab is not None:
				vocab = label_vocab
			else:
				vocab = Vocabulary.from_instances(all_instances)
			all_labels = []
			for instance in all_instances:
				label = instance.fields['label']
				label.index(vocab)
				this_label = label.as_tensor(label.get_padding_lengths())
				all_labels.append(this_label)
			datasets[idx_] = {
								'tokens': all_sentences,
								'labels': np.array(all_labels),
								'pad_idx': tokenizer._tokenizer.pad_token_id,
								'vocab': vocab
							}
		return datasets

	# Setup the classifier for all auxiliary tasks
	def setup_classifiers(self, dropout, embedding_dim, num_layers, ff_multiplier):
		for idx_, dict_ in self.datasets.items():
			self.setup_classifier(dropout, idx_, dict_, embedding_dim, ff_multiplier, num_layers=num_layers)


	def setup_classifier(self, dropout, task_idx, dataset_dict, embedding_dim, ff_multiplier, num_layers=1):
		vocab = dataset_dict['vocab']
		text_field_embedder = self.base_lm_model
		seq2vec_encoder = CLSPooler(embedding_dim)
		hidden_dim = embedding_dim * ff_multiplier
		feedforward = FeedForward(
									embedding_dim, num_layers, hidden_dims=hidden_dim,
									activations=torch.nn.Tanh(), dropout=dropout
								)
		classifier = BasicClassifierWithF1(vocab, text_field_embedder, seq2vec_encoder, feedforward, dropout=dropout, initializer=None)
		classifier.to(self.base_lm_model.device)
		setattr(self, 'AuxHead-{}'.format(task_idx), classifier)
		return classifier

	
# 	def remove_auxiliary_classifiers(self):
# 		param_list = []
# 		for key in self.datasets.keys():
# 			if key == self.primary_task_id:
# 				continue
# 			delattr(self, "AuxHead-{}".format(key))
# 			this_classf = getattr(self, "AuxHead-{}".format(key), None)
# 			assert this_classf is None, 'Auxiliary Classifier {} was not removed'.format(key)

# 	def reinit_primary(self):
# 		prim_classifier = getattr(self, "AuxHead-{}".format(self.primary_task_id), None)
# 		assert prim_classifier is not None, 'Auxiliary Classifier {} not found'.format(self.primary_task_id)
# 		def reset_this(weight, bias):
# 			with torch.no_grad():
# 				init.kaiming_uniform_(weight, a=math.sqrt(5))
# 				if bias is not None:
# 					fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
# 					bound = 1 / math.sqrt(fan_in)
# 					init.uniform_(bias, -bound, bound)

# 		for module in prim_classifier._feedforward_layer._linear_layers:
# 			print('Before : ', module.weight.min().item(), module.bias.min().item())
# 			reset_this(module.weight, module.bias)
# 			print('After : ', module.weight.min().item(), module.bias.min().item())
# 		reset_this(prim_classifier._classification_layer.weight, prim_classifier._classification_layer.bias)

	# Get the list of classifier parameters
	def get_classifier_params(self, keys=None, withbase=False):
		param_list = []
		# Get all the classifier params if keys is not specified
		if keys is None:
			keys = self.datasets.keys()
		for _, key in enumerate(keys):
			this_classf = getattr(self, "AuxHead-{}".format(key), None)
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
			if withbase and key == self.primary_task_id:
				param_list.extend(this_classf.named_parameters())
			else:
				filtered_param_list = [param for pname, param in this_classf.named_parameters() if '_text_field_embedder' not in pname]
				param_list.extend(filtered_param_list)
		return param_list

	# Move the model to the appropriate devices
	def to(self, device):
		for key in self.datasets.keys():
			this_classf = getattr(self, "AuxHead-{}".format(key), None)
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
			this_classf.to(device)
			# Since we have moved this to gpu, we need to re-set the base.
			this_classf._text_field_embedder = self.base_lm_model


	# This sets the optimizer and scheduler for further fine-tuning
	def set_optim(self, optimizer, scheduler):
		# Do this to set the optimizer
		self.optimizer = optimizer
		self.ft_lr_scheduler = scheduler

	# Save the model
	def save(self):
		path = self.save_path
		save_dict = {
				'optimizer_sd': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
				'scheduler': self.ft_lr_scheduler.state_dict() if hasattr(self, 'ft_lr_scheduler') else None,
				'perfs': dict(self.perfs) if hasattr(self, 'perfs') else None,
				'dp_stats': self.dp_stats if hasattr(self, 'dp_stats') else None,
				'weight_stats': self.weight_stats if hasattr(self, 'weight_stats') else None
			}
		for key in self.datasets.keys():
			this_classf = getattr(self, "AuxHead-{}".format(key), None)
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
			save_dict[key] = this_classf.state_dict()
		torch.save(
			save_dict,
			path
		)
	
	def set_save_path(self, save_path):
		self.save_path = save_path

	def load(self):
		# We are assuming that what we care about is the primary task parameters
		primary_classifier = getattr(self, "AuxHead-{}".format(self.primary_task_id), None)
		assert primary_classifier is not None, 'Cannot find primary task classifier head to load'
		state_dict = torch.load(self.save_path)
		for key in self.datasets.keys():
			this_classf = getattr(self, "AuxHead-{}".format(key), None)
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
			this_classf.load_state_dict(state_dict[key])

		if hasattr(self, 'optimizer') and ('optimizer_sd' in state_dict):
			self.optimizer.load_state_dict(state_dict['optimizer_sd'])
			self.ft_lr_scheduler = state_dict['scheduler']
		self.base_lm_model = this_classf._text_field_embedder
	
	def load_primary(self, device):
		# We are assuming that what we care about is the primary task parameters
		primary_classifier = getattr(self, "AuxHead-{}".format(self.primary_task_id), None)
		assert primary_classifier is not None, 'Cannot find primary task classifier head to load'
		state_dict = torch.load(self.save_path)
		primary_classifier.load_state_dict(state_dict[self.primary_task_id])
		primary_classifier.to(device)
		self.base_lm_model = primary_classifier._text_field_embedder

	# Evaluate the classifier
	def evaluate_classifier(self, set_='dev'):
		assert set_ in ['dev', 'test'], 'Wrong dataset specified'
		dataset = None
		if set_ == 'dev':
			fdict_ = {'dev': self.prim_dev_file}
			if self.prim_dev_dataset is None:
				dataset = self.setup_datasets(
										fdict_, self.model_name, self.max_seq_len,
										label_vocab=self.datasets[self.primary_task_id]['vocab']
									)
				self.prim_dev_dataset = dataset['dev']
			dataset = self.prim_dev_dataset
		else:
			fdict_ = {'test': self.prim_test_file}
			if self.prim_test_dataset is None:
				self.prim_test_dataset = self.setup_datasets(
										fdict_, self.model_name, self.max_seq_len,
										label_vocab=self.datasets[self.primary_task_id]['vocab']
									)['test']
			dataset = self.prim_test_dataset

		this_classf = getattr(self, "AuxHead-{}".format(self.primary_task_id), None)
		assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
		# Run the classifier
		torch.cuda.empty_cache()
		# reset the metrics before running new stuff
		try:
			_ = this_classf.get_metrics(reset=True)
		except:
			print('This classifier does not need to reset metrics.')
		this_classf.eval()
		with torch.no_grad():
			for samples in self.dataset_iterator(dataset):
				_ = this_classf(*samples)
		this_classf.train()
		# Get the metrics from the classifier
		return this_classf.get_metrics(reset=True)

	def get_metrics(self, reset=False):
		this_classf = getattr(self, "AuxHead-{}".format(self.primary_task_id), None)
		assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
		# Get the metrics from the classifier
		return this_classf.get_metrics(reset=reset)

	# Get samples for a task
	def get_classifier_samples(self, task_id, nsamples, dataset=None):
		if dataset is None:
			dataset = self.datasets[task_id]
		num_egs = len(dataset['tokens'])
		idxs = np.random.choice(num_egs, size=nsamples, replace=False)
		sentences, labels = [dataset['tokens'][i] for i in idxs], dataset['labels'][idxs]
		sentences = collate(sentences, dataset['pad_idx'])
		sentences = sentences.to(self.base_lm_model.device)
		labels = torch.IntTensor(labels).to(sentences.device)
		return sentences, labels
	
	# This function should set the dev head
	def set_dev_head(self):
		if self.prim_dev_dataset is None:
			fdict_ = {'dev': self.prim_dev_file}
			self.prim_dev_dataset = self.setup_datasets(
										fdict_, self.model_name, self.max_seq_len,
										label_vocab=self.datasets[self.primary_task_id]['vocab']
									)['dev']
		# Learn the head here
		this_classf = self.learn_dev_head()
		# Get the dev gradient here
		dev_sent, dev_labels = self.get_classifier_samples(self.primary_task_id, self.batch_sz, dataset=self.prim_dev_dataset)
		loss_ = this_classf(dev_sent, dev_labels)['loss']
		gradients = torch.autograd.grad(loss_, this_classf.parameters(), allow_unused=True)
		return gradients
	
	# This function resets the dev-head. We use this anytime we need a new approximation of the dev head
	def reset_dev_head(self):
		dev_head_name = "AuxHead-{}-{}".format('dev', self.primary_task_id)
		this_classf = getattr(self, dev_head_name, None)
		assert this_classf is not None, 'The dev head has to already be instantiated for this function to be called'
		del this_classf
		setattr(self, dev_head_name, None)

	def learn_dev_head(self):
		assert hasattr(self, 'options'), 'The options need to be set for training of the dev head'
		this_classf = getattr(self, "AuxHead-{}-{}".format('dev', self.primary_task_id), None)
		head_name =  "{}-{}".format('dev', self.primary_task_id)
		dev_params = None
		if this_classf is None:
			# Need to instantiate the classifier head
			this_classf = self.setup_classifier(
								self.dropout, head_name, self.prim_dev_dataset,
								self.embedding_dim, self.ff_multiplier, num_layers=self.num_layers
							)
			# Setup optimizer for dev head
			dev_params = self.get_classifier_params([head_name], withbase=False)
			dev_optim =  AdamW(
									dev_params, betas=eval(self.options.classf_betas),
									weight_decay=self.options.classf_dev_wd, lr=self.options.classf_dev_lr
								)
		else:
			return this_classf # We train this once and re-use
		
		# This is the first time instantiating this head so we need to train it
		assert dev_optim is not None, 'The optimizer for the dev head has not been instantiated'

		# perform gradient descent to get the dev-head
		samples = self.get_classifier_samples(self.primary_task_id, self.batch_sz)
		assert dev_params is not None, 'Dev Params should have been instantiated above'
		prev_loss_, tol = 1e10, 1e-3
		for i in range(self.options.classf_ft_iters):
			output = this_classf(*samples)
			loss_ = output['loss']
			# This ensures that we only train the dev-head and keep the body fixed
			grads = torch.autograd.grad(loss_, dev_params, allow_unused=True)
			for p, g in zip(dev_params, grads):
				assert g is not None, 'This should have a gradient'
				p.grad = g
			dev_optim.step()
			metrics = this_classf.get_metrics(reset=True)
# 			f1, acc = metrics['f1'], metrics['accuracy']
			if abs(loss_ - prev_loss_) < tol:
				break
			prev_loss_ = loss_.item()
			del grads
			torch.cuda.empty_cache()
		return this_classf
	
	def set_mlm_grads(self, grads):
		if grads is not None:
			assert self.MLM_grads is None, 'Need to make sure grads are none before setting'
		else:
			assert self.MLM_grads is not None, 'Need to make sure grads are set before setting to none'
		self.MLM_grads = grads

	# Calculate the gradient for the classifier and lm
	def classifier_sample_grad(self):
		gradient_dict = None
		if self.alpha_generator_algo.is_meta:
			# Do all the setup here
			gradient_dict = {}
			# Get the gradients w.r.t the different tasks
			for task_id in self.base_task_dataset_files.keys():
				this_classf = getattr(self, "AuxHead-{}".format(task_id), None)
				assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(task_id)
				sent_dict, labels = self.get_classifier_samples(task_id, self.batch_sz)
				loss_ = this_classf(sent_dict, labels)['loss']
				gradients = torch.autograd.grad(loss_, this_classf.parameters(), allow_unused=True)
				gradient_dict[task_id] = gradients
			if not hasattr(self, 'body_params_end'):
				self.body_params_end = get_body_end(this_classf)
			
			# Get the MLM current gradient. Double check that the MLM gradient is set correctly
			assert self.MLM_grads is not None, 'MLM Grads should have been set by now'
			gradient_dict["MLM"] = self.MLM_grads[:self.body_params_end]

			# Get the gradient dictionary
			gradient_dict["dev-{}".format(self.primary_task_id)] = self.set_dev_head()
			dev_task_grads = gradient_dict["dev-{}".format(self.primary_task_id)][:self.body_params_end]
			meta_weights = self.alpha_generator_algo.weights
			with torch.no_grad():
				dev_norm = calc_norm(dev_task_grads)
				all_tasks = self.aux_tasks + [self.primary_task_id]
				for task_id in all_tasks:
					task_norm = calc_norm(gradient_dict[task_id][:self.body_params_end])
					cos_sim = dot_prod(dev_task_grads, gradient_dict[task_id][:self.body_params_end])
					cos_sim = cos_sim / (dev_norm * task_norm)
					self.dp_stats[task_id].append(cos_sim)
					cos_sim = (torch.zeros_like(meta_weights[task_id]) - cos_sim)  / self.grad_accum_factor
					if meta_weights[task_id].grad is None:
						meta_weights[task_id].grad = cos_sim
					else:
						meta_weights[task_id].grad.add_(cos_sim)
					self.weight_stats[task_id].append((meta_weights[task_id].item(), meta_weights[task_id].grad.item(), dev_norm, task_norm, self.dp_stats[task_id][-1]))
				print('\n')


		total_loss = 0.0
		for task_id in self.base_task_dataset_files.keys():
			this_classf = getattr(self, "AuxHead-{}".format(task_id), None)
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(task_id)
			if gradient_dict is None:
				sent_dict, labels = self.get_classifier_samples(task_id, self.batch_sz)
				output_dict = this_classf(sent_dict, labels)
				total_loss = total_loss + self.alpha_generator_algo[task_id]*output_dict['loss']
				total_loss = total_loss / self.grad_accum_factor
				total_loss.backward()
			else:
				gradients = gradient_dict[task_id]
				with torch.no_grad():
					scaling = self.alpha_generator_algo[task_id] / self.grad_accum_factor
					for idx, (p, g) in enumerate(zip(this_classf.parameters(), gradients)):
						if p.grad is None:
							p.grad = torch.zeros_like(p)
						if g is None:
							continue
						p.grad.add_(scaling * g)

	# We train the primary head. This is further finetuning on top pre-training
	def train_primary(self, n_iters, optimizer, lr_scheduler, max_grad_norm, patience=3):
		# Setup Optimizer and stuff
		best_iter = 0
		if self.prim_train_dataset is None:
			fdict_ = {'train': self.base_task_dataset_files[self.primary_task_id]}
			dataset = self.setup_datasets(
										fdict_, self.model_name, self.max_seq_len,
										label_vocab=self.datasets[self.primary_task_id]['vocab']
									)
			self.prim_train_dataset = dataset['train']
		dataset = self.prim_train_dataset
		prim_classf = getattr(self, "AuxHead-{}".format(self.primary_task_id), None)
		prim_classf.train()
		assert prim_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
		self.perfs = defaultdict(list)
		iters_since_improvement = 0
		for iter_ in range(n_iters):
			print('Currently on Classifier Epoch {}/{}'.format(iter_ + 1, n_iters))
			iterator = self.dataset_iterator(dataset, shuffle=True)
			total_iters = len(dataset['tokens']) // self.batch_sz + 1
			# Get the primary classifier
			iterator = tqdm(iterator, total= total_iters, desc="Classifier Train Iterator")
			for idx, samples in enumerate(iterator):
				if (idx + 1) % self.grad_accum_factor == 0:
					# We want to take a gradient step
					torch.nn.utils.clip_grad_norm_(prim_classf.parameters(), max_grad_norm)
					optimizer.step()
					if lr_scheduler is not None:
						lr_scheduler.step()
					optimizer.zero_grad()
				output_dict = prim_classf(*samples)
				total_loss = output_dict['loss'] / self.grad_accum_factor
				total_loss.backward()
			# We want to evaluate the classifier
			train_metrics = self.get_metrics(reset=True)
			dev_metrics = self.evaluate_classifier(set_='dev')
			test_metrics = self.evaluate_classifier(set_='test')
			# Report the metrics
			for k, v in train_metrics.items():
				to_show = k, v, dev_metrics[k], test_metrics[k]
				print_out = "[{}] | Train : {:.3f} | Dev Set : {:.3f} | Test Set : {:.3f}".format(*to_show)
				print(print_out)
			self.perfs['train'].append((train_metrics['f1'], train_metrics['accuracy']))
			self.perfs['dev'].append((dev_metrics['f1'], dev_metrics['accuracy']))
			self.perfs['test'].append((test_metrics['f1'], test_metrics['accuracy']))
			if dev_metrics['f1'] >= self.perfs['dev'][best_iter][0]:
				best_iter = iter_
				iters_since_improvement = 0
			else:
				iters_since_improvement += 1
				if iters_since_improvement >= patience:
					print('Breaking because we have no improvement in {} epochs'.format(patience))
					break
		best_f1, best_acc = self.perfs['test'][best_iter]
		return best_f1, best_acc, self.perfs


	def dataset_iterator(self, dataset, shuffle=False):
		total_egs = len(dataset['tokens'])
		# +1 is so as to not have an off-by-1 bug
		num_batches = total_egs // self.batch_sz + 1
		if shuffle:
			idxs = np.random.permutation(total_egs)
		else:
			idxs = list(range(total_egs))
		for i in range(num_batches):
			this_idxs = idxs[(i * self.batch_sz): ((i + 1) * self.batch_sz)]
			sentences = [dataset['tokens'][id_] for id_ in this_idxs]
			labels = dataset['labels'][this_idxs]
			sentences = collate(sentences, dataset['pad_idx'])
			sentences = sentences.to(self.base_lm_model.device)
			labels = torch.IntTensor(labels).to(self.base_lm_model.device)
			yield sentences, labels
			del sentences
			del labels


	# Hack this models forward pass to respond to the lm forward pass
	def forward(*args, **kwargs):
		return self.base_lm_model(*args, **kwargs)

