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

# Setting up gpu usage monitoring
nvidia_smi.nvmlInit()
gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())


def print_gpu_stats():
	mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
	print('Gpu mem: {} (GiB)'.format(mem_res.used / (1024**2)))  # usage in GiB
	print('Gpu used: {} %'.format(100 * (mem_res.used / mem_res.total)))


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
					alpha_generator_algo='equal',
					primary_task_id=None,
					batch_sz=8,
					samples_per_run=1,
					prim_dev_file =None, 
					prim_test_file =None,
					grad_accum_factor=8
	):
		assert save_path is not None, 'Invalid Save Path Provided for Classifier Head'
		assert isinstance(base_task_dataset_files, dict), 'Invalid type of base_task_dataset_files. Expected array'
		assert primary_task_id is not None, 'No primary task id is given'
		assert primary_task_id in base_task_dataset_files, 'Primary Task not included in the list of dataset files'
		self.save_path = save_path
		self.base_lm_model = base_lm_model
		self.base_task_dataset_files = base_task_dataset_files
		self.datasets = self.setup_datasets(base_task_dataset_files, model_name, max_seq_len, lazy=False)

		self.setup_classifiers(dropout, embedding_dim, num_layers, ff_multiplier)
		self.samples_per_run = samples_per_run
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


	def setup_alpha_generator(self, options):
		# Create a special task called MLM.
		task_names = list(self.base_task_dataset_files.keys())
		task_names.append("MLM")
		# Remove the primary task name
		aux_tasks = [x for x in task_names if x != self.primary_task_id]
		self.alpha_generator_algo = get_alpha_generator(options, self.primary_task_id, aux_tasks)


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
			vocab = dict_['vocab']
			text_field_embedder = self.base_lm_model
			seq2vec_encoder = CLSPooler(embedding_dim)
			hidden_dim = embedding_dim * ff_multiplier
			feedforward = FeedForward(
										embedding_dim, num_layers, hidden_dims=hidden_dim,
										activations=torch.nn.Tanh(), dropout=dropout
									)
			classifier = BasicClassifierWithF1(vocab, text_field_embedder, seq2vec_encoder, feedforward, dropout=dropout)
			setattr(self, 'AuxHead-{}'.format(idx_), classifier)
# 			self.add_module('AuxHead-{}'.format(idx_), classifier)

	# Get the list of classifier parameters
	def get_classifier_params(self, keys=None):
		param_list = []
		if keys is None:
			keys = self.datasets.keys()
		for key in keys:
			this_classf = getattr(self, "AuxHead-{}".format(key), None)
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
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


	def set_optim(self, optimizer, scheduler):
		# Do this to set the optimizer
		self.optimizer = optimizer
		self.ft_lr_scheduler = scheduler

	def save(self):
		path = self.save_path
		save_dict = {
				'optimizer_sd': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
				'scheduler': self.ft_lr_scheduler.state_dict() if hasattr(self, 'ft_lr_scheduler') else None,
				'perfs': dict(self.perfs) if hasattr(self, 'perfs') else None
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

	def remove_auxiliary_classifiers(self):
		param_list = []
		for key in self.datasets.keys():
			if key == self.primary_task_id:
				continue
			delattr(self, "AuxHead-{}".format(key))
			this_classf = getattr(self, "AuxHead-{}".format(key), None)
			assert this_classf is None, 'Auxiliary Classifier {} was not removed'.format(key)


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
		with torch.no_grad():
			for samples in self.dataset_iterator(dataset):
				_ = this_classf(*samples)
		# Get the metrics from the classifier
		return this_classf.get_metrics(reset=True)

	def get_metrics(self, reset=False):
		this_classf = getattr(self, "AuxHead-{}".format(self.primary_task_id), None)
		assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
		# Get the metrics from the classifier
		return this_classf.get_metrics(reset=reset)

	# Get samples for a task
	def get_classifier_samples(self, task_id, nsamples):
		num_egs = len(self.datasets[task_id]['tokens'])
		idxs = np.random.choice(num_egs, size=nsamples, replace=False)
		sentences, labels = [self.datasets[task_id]['tokens'][i] for i in idxs], self.datasets[task_id]['labels'][idxs]
		sentences = collate(sentences, self.datasets[task_id]['pad_idx'])
		sentences = sentences.to(self.base_lm_model.device)
		labels = torch.IntTensor(labels).to(sentences.device)
		return sentences, labels

	# Calculate the gradient for the classifier and lm
	def classifier_sample_grad(self):
		# Account for the masked-language-modelling task
		mlm_weight = self.alpha_generator_algo["MLM"]
		# Modify the current gradients in the base lm model
		with torch.no_grad():
			for pname, p in self.base_lm_model.named_parameters():
				assert p.grad is not None, 'Base LM Model should not have None grad for {}'.format(pname)
				p.grad.mul_(mlm_weight)

		total_loss = 0.0
		for task_id in self.base_task_dataset_files.keys():
			this_classf = getattr(self, "AuxHead-{}".format(task_id), None)
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(task_id)
			for _ in range(self.samples_per_run):
				sent_dict, labels = self.get_classifier_samples(task_id, self.batch_sz)
				output_dict = this_classf(sent_dict, labels)
				total_loss = total_loss + self.alpha_generator_algo[task_id]*output_dict['loss']
				total_loss = total_loss / self.grad_accum_factor
				total_loss.backward()
				
	
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
		assert prim_classf is not None, 'Auxiliary Classifier {} not found'.format(key)
		self.perfs = defaultdict(list)
		iters_since_improvement = 0
		for iter_ in range(n_iters):
			print('Currently on Classifier Epoch {}/{}'.format(iter_ + 1, n_iters))
			iterator = self.dataset_iterator(dataset, shuffle=True)
			total_iters = len(dataset['tokens']) // self.batch_sz
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
		num_batches = total_egs // self.batch_sz
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

