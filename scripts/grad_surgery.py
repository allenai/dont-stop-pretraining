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
)
import nvidia_smi
import sys
PATH="/home/ldery/internship/dsp/dont_stop_pretraining"
# PATH="/home/jupyter/projects/dsp/dont_stop_pretraining"
sys.path.insert(1, PATH)
from models import BasicClassifierWithF1
from data.dataset_readers.text_classification_json_reader_with_sampling import TextClassificationJsonReaderWithSampling
from modules.seq2vec_encoders.cls_pooler import CLSPooler
PATH="/home/ldery/internship/meta4da/algorithms"
sys.path.insert(1, PATH)
from utils import *
import pdb

# Object for grad weights
GradWeights = namedtuple('GradWeights', 'eta_tilde eta_pos eta_neg')

# Setting up gpu usage monitoring
nvidia_smi.nvmlInit()
gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())


def print_gpu_stats():
	mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
	print('Gpu mem: {} (GiB)'.format(mem_res.used / (1024**2)))  # usage in GiB
	print('Gpu used: {} %'.format(100 * (mem_res.used / mem_res.total)))


class ModelWithGradSurgery(AutoModelWithLMHead):
	''' Wrapper around a basic model so we can do gradient surgery on the model'''
	def __init__(
					self,
					model_name,
					base_lm_model,
					base_task_dataset_file,
					step_frequency=1,
					max_seq_len=512,
					pca_every=10,
					num_basis=10,
					num_samples_for_basis=64,
					eta_set=(1.0, 1.0, -1.0),
					dropout=0.0,
					embedding_dim=768,
					ff_multiplier=1,
					num_layers=1,
					num_subspace_decomp_layers=-2,  # negative means we take the last n layers
					test_task_file=None,
					dev_task_file=None,
					max_norm=1.0,
					classf_train_batch_sz=128,
					save_path=None,
					multitask_weight=None
	):
		assert save_path is not None, 'Invalid Save Path Provided for Classifier Head'
		self.save_path = save_path
		self.base_lm_model = base_lm_model
		self.base_task_dataset_file = base_task_dataset_file
		self.dataset = self.setup_dataset(base_task_dataset_file, model_name, max_seq_len, lazy=False)
		self.dev_task_file, self.dev_dataset = dev_task_file, None
		self.test_task_file, self.test_dataset = test_task_file, None

		self.classifier = self.setup_classifier(dropout, embedding_dim, num_layers, ff_multiplier)
		self.subspace_decomp_layers = self.get_subspace_layers(num_subspace_decomp_layers)
		self.subspace_param_list = tuple([v for k, v in self.classifier.named_parameters() if self.is_subspace_layer(k)])
		self.pca_cntr = 0
		self.pca_every = pca_every
		self.g_weights = GradWeights(
										eta_tilde=eta_set[0], eta_pos=eta_set[1],
										eta_neg=eta_set[2]
									)
		self.subspace_nsamples = num_samples_for_basis  # Assuming this is set to max-out memory
		self.samples_per_batch = self.subspace_nsamples // 2
		self.num_subspace_basis = num_basis
		self.max_norm = 1.0
		self.max_seq_len = max_seq_len
		self.model_name = model_name
		self.classf_train_batch_sz = classf_train_batch_sz
		self.classf_ft_batch_sz = classf_train_batch_sz
		self.multitask_weight = multitask_weight

	def setup_dataset(self, base_task_dataset_file, model_name, max_seq_len, label_vocab=None, lazy=False):
		# Instantiate dataset reader
		indexers = {'tokens': PretrainedTransformerIndexer(model_name, do_lowercase=False)}
		tokenizer = PretrainedTransformerTokenizer(model_name, do_lowercase=False, start_tokens=["<s>"], end_tokens=["</s>"])
		dataset_reader = TextClassificationJsonReaderWithSampling(
							token_indexers=indexers, tokenizer=tokenizer,
							max_sequence_length=max_seq_len, lazy=lazy
						)
		# Read from the dataset
		all_samples = dataset_reader._read(base_task_dataset_file)
		pretrain_vocab = tokenizer._tokenizer.encoder
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
		return {
					'tokens': all_sentences,
					'labels': np.array(all_labels),
					'pad_idx': tokenizer._tokenizer.pad_token_id,
					'vocab': vocab
				}

	def get_subspace_layers(self, num_layers):
		all_layer_names = []
		for k, _ in self.classifier.named_parameters():
			all_layer_names.append(k)
		layer_nums, layer_names = [], defaultdict(list)
		for id_ in all_layer_names:
			attrs = id_.split('.')
			if 'layer' in attrs:
				idx = attrs.index('layer')
				layer_nums.append(int(attrs[idx + 1]))
				layer_names[int(attrs[idx + 1])].append(id_)
		sorted_layers = sorted(list(set(layer_nums)))
		if num_layers < 0:
			chosen = sorted_layers[num_layers:]
		else:
			chosen = sorted_layers[:num_layers]
		result = []
		for k in all_layer_names:
			if 'embeddings' in k:
				result.append(k)
		layer_results = [y for x in chosen for y in layer_names[x]]
		result.extend(layer_results)
		total_size = 0
		for k, v in self.classifier.named_parameters():
			if k in result:
				total_size += v.numel()
		print('THIS IS THE TOTAL SIZE !!!! - ', total_size / 1e6)
		return result

	def is_subspace_layer(self, layer_name):
		for k in self.subspace_decomp_layers:
			if layer_name in k:
				return True
		return False

	def setup_classifier(self, dropout, embedding_dim, num_layers, ff_multiplier):
		vocab = self.dataset['vocab']
		text_field_embedder = self.base_lm_model
		seq2vec_encoder = CLSPooler(embedding_dim)
		hidden_dim = embedding_dim * ff_multiplier
		feedforward = FeedForward(
									embedding_dim, num_layers, hidden_dims=hidden_dim,
									activations=torch.nn.Tanh(), dropout=dropout
								)
		return BasicClassifierWithF1(vocab, text_field_embedder, seq2vec_encoder, feedforward, dropout=dropout)

	def to(self, device):
		self.base_lm_model.to(device)
		self.classifier.to(device)
		# Since we have moved this to gpu, we need to re-set the base.
		self.classifier._text_field_embedder = self.base_lm_model
		# the tensors have been moved so use those
		self.subspace_param_list = tuple([v for k, v in self.classifier.named_parameters() if self.is_subspace_layer(k)])

	def set_optim(self, optimizer, scheduler):
		# Do this to set the optimizer
		self.optimizer = optimizer
		self.ft_lr_scheduler = scheduler

	def save(self):
		path = self.save_path
		torch.save(
			{
				'classifier_sd': self.classifier.state_dict(),
				'optimizer_sd': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
				'scheduler': self.ft_lr_scheduler.state_dict() if hasattr(self, 'ft_lr_scheduler') else None,
				'dataset_files': {
									'train': self.base_task_dataset_file,
									'dev': self.dev_task_file,
									'test': self.test_task_file,
								},
				'decomp_layers': self.subspace_decomp_layers,
				'perfs': dict(self.perfs) if hasattr(self, 'perfs') else None},
			path
		)

	def load(self):
		state_dict = torch.load(self.save_path)
		self.classifier.load_state_dict(state_dict['classifier_sd'])
		if hasattr(self, 'optimizer') and ('optimizer_sd' in state_dict):
			self.optimizer.load_state_dict(state_dict['optimizer_sd'])
			self.ft_lr_scheduler = state_dict['scheduler']
		self.base_lm_model = self.classifier._text_field_embedder

	def get_ft_steps(self, num_epochs):
		# [NB] I am using self.subspace_nsamples as a proxy for the instantaneous batch size
		# Since self.subspace_nsamples is chosen to max out gpu
		grad_accum_steps = self.classf_ft_batch_sz // self.subspace_nsamples
		return len(self.dataset['tokens']) // grad_accum_steps * num_epochs

	def evaluate_classifier(self, set_='dev'):
		assert set_ in ['dev', 'test'], 'Wrong dataset specified'
		dataset = None
		if set_ == 'dev':
			if self.dev_dataset is None:
				self.dev_dataset = self.setup_dataset(
										self.dev_task_file, self.model_name, self.max_seq_len,
										label_vocab=self.dataset['vocab']
									)
			dataset = self.dev_dataset
		else:
			if self.test_dataset is None:
				self.test_dataset = self.setup_dataset(
										self.test_task_file, self.model_name, self.max_seq_len,
										label_vocab=self.dataset['vocab']
									)
			dataset = self.test_dataset
		self.classifier.eval()
		# Run the classifier
		for samples in self.dataset_iterator(dataset):
			_ = self.classifier(*samples)
		self.classifier.train()
		# Get the metrics from the classifier
		return self.classifier.get_metrics(reset=True)

	def do_finetuning(self, max_ft_steps, logger, patience=3):
		assert self.optimizer is not None, 'Optimizer not given for finetuning'
		# Create the ft scheduler here
		self.classifier.train()
		self.perfs = defaultdict(lambda: defaultdict(list))
		dev_metrics = self.evaluate_classifier(set_='dev')
		logger.info('About to start finetuning model')
		logger.info('Metrics before Finetuning : ' + str(dev_metrics))
		for iter_ in range(max_ft_steps):
			rounds_cntr = 0
			# [NB] I am using self.subspace_nsamples as a proxy for the instantaneous batch size
			# Since self.subspace_nsamples is chosen to max out gpu
			grad_accum_steps = self.classf_ft_batch_sz // self.samples_per_batch
			for samples in self.dataset_iterator(self.dataset, shuffle=True):
				loss = self.classifier(*samples)['loss'] / grad_accum_steps
				loss.backward()
				rounds_cntr += 1
				if rounds_cntr == grad_accum_steps:
					rounds_cntr = 0
					clip_grad_norm_(self.classifier.parameters(), self.max_norm)
					self.optimizer.step()
					self.optimizer.zero_grad()
			train_metrics = self.classifier.get_metrics(reset=True)
			dev_metrics = self.evaluate_classifier(set_='dev')
			test_metrics = self.evaluate_classifier(set_='test')
			for k, v in train_metrics.items():
				info = iter_, k, v, dev_metrics[k], test_metrics[k]
				print_out = "Epoch : {} | {} | Train : {:.3f} | Dev Set : {:.3f} | Test Set : {:.3f}".format(*info)
				logger.info(print_out)
				self.perfs['train'][k].append(v)
				self.perfs['test'][k].append(test_metrics[k])
				self.perfs['dev'][k].append(dev_metrics[k])
			self.ft_lr_scheduler.step(dev_metrics['f1'])
			if len(self.perfs['dev'][k]) <= patience:
				self.save()
				continue
			abs_max_, p_max = max(self.perfs['dev']['f1']), max(self.perfs['dev']['f1'][-patience:])
			if p_max < abs_max_:
				# We have not improved in patience # of epochs. Time to break out of loop
				best_epoch = np.argmax(self.perfs['dev']['f1'])
				print_out = "Best Dev Epoch : {} | Dev F1 : {}".format(best_epoch, abs_max_)
				logger.info(print_out)
				test_info = ('F1', self.perfs['test']['f1'][best_epoch], 'Accuracy', self.perfs['test']['accuracy'][best_epoch])
				logger.info("Test Stats for Epoch : {} = {:.3f} | {} = {:.3f}".format(*test_info))
				train_info = ('F1', self.perfs['train']['f1'][best_epoch], 'Accuracy', self.perfs['train']['accuracy'][best_epoch])
				logger.info("Train Stats for Epoch : {} = {:.3f} | {} = {:.3f}".format(*train_info))
				break
			else:
				self.save()
		self.load()
		logger.info('Done Finetuning Our Model')

	def get_classifier_samples(self, nsamples):
		num_egs = len(self.dataset['tokens'])
		idxs = np.random.choice(num_egs, size=self.subspace_nsamples, replace=False)
		sentences, labels = [self.dataset['tokens'][i] for i in idxs], self.dataset['labels'][idxs]
		sentences = collate(sentences, self.dataset['pad_idx'])
		sentences = sentences.to(self.subspace_param_list[0].device)
		labels = torch.IntTensor(labels).to(self.subspace_param_list[0].device)
		return sentences, labels

	def dataset_iterator(self, dataset, shuffle=False):
		total_egs = len(dataset['tokens'])
		num_batches = total_egs // self.samples_per_batch
		if shuffle:
			idxs = np.random.permutation(total_egs)
		else:
			idxs = list(range(total_egs))
		for i in range(num_batches):
			this_idxs = idxs[(i * self.samples_per_batch): ((i + 1) * self.samples_per_batch)]
			sentences = [dataset['tokens'][id_] for id_ in this_idxs]
			labels = dataset['labels'][this_idxs]
			sentences = collate(sentences, dataset['pad_idx'])
			sentences = sentences.to(self.subspace_param_list[0].device)
			labels = torch.IntTensor(labels).to(self.subspace_param_list[0].device)
			yield sentences, labels

	# Make our model functional w.r.t the subspace_decomp_layers
	def functional_(self, data_tuple):
		def fn(*params):
			for name, p in zip(self.subspace_decomp_layers, params):
				set_attr(self.classifier, name.split("."), p)
			result = self.classifier(*data_tuple)['loss_full']
			return result
		return fn

	def make_functional(self):
		class_ = list(self.classifier.named_parameters())
		for name, p in class_:
			if not self.is_subspace_layer(name):
				continue
			# strip the model of these attributes so it's purely functional
			del_attr(self.classifier, name.split("."))

	# Hack this models forward pass to respond to the lm forward pass
	def forward(*args, **kwargs):
		return self.base_lm_model(*args, **kwargs)

	def get_subspace_layer_grads(self):
		grad_vec = []
		for name, p_ in self.base_lm_model.named_parameters():
			if self.is_subspace_layer(name):
				assert p_.grad is not None, 'Cannot do subspace decomp on layer with no grad'
				grad_vec.append(p_.grad)
		_, grad_vec = vectorize(grad_vec)
		return grad_vec

	def set_subspace_layer_grads(self, grad_vec):
		cur_pos = 0
		with torch.no_grad():
			for name, p_ in self.base_lm_model.named_parameters():
				if self.is_subspace_layer(name):
					if p_.grad is None:
						raise ValueError('Encountered a param with None gradient')
					assert p_.grad is not None, 'Cannot do subspace decomp on layer with no grad'
					numel = p_.grad.numel()
					to_copy = grad_vec[cur_pos: (cur_pos + numel)].view(p_.grad.shape)
					p_.grad.data.copy_(to_copy)
					cur_pos += numel
		assert cur_pos == grad_vec.shape[0], 'Not all parameters were used up. Something is wrong'

	def classifier_sample_grad(
				self, set_classifier_grad=True, set_lm_grad=False, classf_grad_weight=1.0, lm_grad_weight=0.0
			):
		sent_dict, labels = self.get_classifier_samples(self.samples_per_batch)
		output_dict = self.classifier(sent_dict, labels)
		loss = output_dict['loss']
		params_w_grads_to_gather = []
		params_w_grads_to_gather.extend(self.subspace_param_list)
		names = list(self.subspace_decomp_layers)
		n_subspace = len(self.subspace_param_list)
		if set_classifier_grad:
			for k, v in self.classifier.named_parameters():
				if '_text_field_embedder' not in k:
					params_w_grads_to_gather.append(v)
					names.append(k)
			grads = torch.autograd.grad(loss, params_w_grads_to_gather, allow_unused=True)
			for grad, p_ in zip(grads[n_subspace:], params_w_grads_to_gather[n_subspace:]):
				# Set the gradients of the classifer components here
				if p_.grad is None:
					p_.grad = torch.zeros_like(p_)
				p_.grad.data.add_(grad * classf_grad_weight)
		subspace_grads = grads[:n_subspace]
		if set_lm_grad:
			# Setting the transformer layer grads - during multitasking
			for grad, p_ in zip(subspace_grads, self.subspace_param_list):
				if p_.grad is None:
					p_.grad = torch.zeros_like(p_)
				p_.grad.data.add_(grad * lm_grad_weight)
		_, grad_vec = vectorize(subspace_grads)
		return grad_vec

	def do_multitask_backward(self):
		num_iters = self.classf_train_batch_sz // self.samples_per_batch
		for i in range(num_iters):  # We are doing gradient accumulation for a better gradient estimate
			self.classifier_sample_grad(set_classifier_grad=True, set_lm_grad=True, lm_grad_weight=self.multitask_weight)

	def get_projgrad(self, grad_vec, ortho_basis, mt_weighting=0.0):
		# Get the low-rank approx grad for classifier task
		num_iters = self.classf_train_batch_sz // self.samples_per_batch
		classf_grad_weight = (1.0 if mt_weighting == 0 else mt_weighting) / num_iters
		rand_classifier_grad = None
		for i in range(num_iters):  # We are doing gradient accumulation for a better gradient estimate
			this_grad = self.classifier_sample_grad(set_classifier_grad=True, classf_grad_weight=classf_grad_weight).unsqueeze(0)
			with torch.no_grad():
				if rand_classifier_grad is None:
					rand_classifier_grad = this_grad / num_iters
				else:
					rand_classifier_grad.add_(this_grad / num_iters)
		rand_classifier_lowrank = rand_classifier_grad.matmul(ortho_basis.t())
		with torch.no_grad():
			# project unto ortho basis
			low_rank = grad_vec.unsqueeze(0).matmul(ortho_basis.t())  # Get the low-rank rep of the lm task
			# low rank components where validation and train agree in direction
			mask_pos = ((low_rank * rand_classifier_lowrank) > 0.0).float()

			in_span_grad_pos = (low_rank * mask_pos).matmul(ortho_basis).squeeze()
			in_span_grad_neg = (low_rank * (1.0 - mask_pos)).matmul(ortho_basis).squeeze()
			out_span_grad = grad_vec - (in_span_grad_pos + in_span_grad_neg)

			# Calculate the new gradient
			out_span_grad.mul_(self.g_weights.eta_tilde)
			in_span_grad_pos.mul_(self.g_weights.eta_pos)
			in_span_grad_neg.mul_(self.g_weights.eta_neg)
			new_grads = out_span_grad + in_span_grad_neg + in_span_grad_pos + (mt_weighting * rand_classifier_grad.squeeze())
		return new_grads

	def rand_sample_ortho_grad_basis(self):
		proj_mat = torch.normal(
									mean=0.0, std=1.0, size=(self.subspace_nsamples, self.num_subspace_basis),
									device=self.subspace_param_list[0].device
								)
		# We may use a subset of the reference set
		base_set = self.get_classifier_samples(self.subspace_nsamples)
		prod_ = []
		# Turn off dropout for better estimate of subspace
		for name, child in self.classifier.named_modules():
			if 'dropout' in name:
				child.eval()
		for i in range(self.num_subspace_basis):
			v_ = proj_mat[:, i]
			# emptying cache to pre-empt memory issues. Might want to wrap in a try-catch-block
			torch.cuda.empty_cache()
			self.make_functional()
			out = torch.autograd.functional.vjp(self.functional_(base_set), self.subspace_param_list, strict=True, v=v_)
			prod_.append(vectorize(out[1])[1].unsqueeze(1))
		# Turn dropout back on
		for name, child in self.classifier.named_modules():
			if 'dropout' in name:
				child.train()
		with torch.no_grad():
			prod_ = torch.cat(prod_, dim=1)
			# My implementation of qr turned out to be faster
			# Seems the torch version makes underlying api calls which take time.
			ortho_basis = my_qr(prod_.t())
		# Now undo harm from making the model functional w.r.t chosen params :
		for name, p in zip(self.subspace_decomp_layers, self.subspace_param_list):
			set_attr(self.classifier, name.split("."), p)
		return ortho_basis

	def set_surrogate_gradient(self, max_grad_norm=None):
		# print('GPU Stats before Doing Alignment')
		# print_gpu_stats()
		lm_grad_vec = self.get_subspace_layer_grads()
		self.pca_cntr += 1
		should_optimize_classifier = False
		if (self.pca_every == self.pca_cntr) or (not hasattr(self, 'ortho_basis')):
			self.pca_cntr = 0
			should_optimize_classifier = True
			self.ortho_basis = self.rand_sample_ortho_grad_basis()
		mt_weighting = self.multitask_weight if self.multitask_weight is not None else 0.0
		projected_grad = self.get_projgrad(lm_grad_vec, self.ortho_basis, mt_weighting=mt_weighting)
		self.set_subspace_layer_grads(projected_grad)
		if should_optimize_classifier:
			for k, v in self.classifier.named_parameters():
				if '_text_field_embedder' not in k:
					assert v.grad is not None, 'Param - {} has no gradient'.format(k)
					clip_grad_norm_(v.grad, self.max_norm)
		# print('Gpu stats after Doing Alignment')
		# print_gpu_stats()
		return should_optimize_classifier
