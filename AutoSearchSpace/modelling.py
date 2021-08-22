import torch
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.modules import FeedForward
from transformers import (
	AutoModel,
	AdamW,
	AutoConfig,
)
from tqdm import tqdm, trange
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from allennlp.data import Vocabulary
from collections import defaultdict
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import os
import sys
import math
PATH="/home/ldery/projects/AutoAuxiliaryLoss/dont_stop_pretraining/"
sys.path.insert(1, PATH)
from models import BasicClassifierWithF1, BasicSequenceTagger
from modules.seq2vec_encoders.cls_pooler import CLSPooler
from dataset.dataset_readers.text_classification_json_reader_with_sampling import TextClassificationJsonReaderWithSampling

try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter

# Calculates the dot products of 2 gradient vectors
def dot_prod(g1, g2, ppdp=None):
	total = 0.0
	for p1, p2 in zip(g1, g2):
		if p1 is None or p2 is None:
			continue
		sum_ = (p1 * p2).sum()
		total += sum_
		# Added this so we can do analysis on per-parameter dot-products
		if ppdp is not None:
			norm1, norm2 = torch.norm(p1), torch.norm(p2)
			ppdp.append((sum_/(norm1 * norm2)).item())
	total = total.item() if isinstance(total, torch.Tensor) else total
	return total

# Calculates the norm of a list of vectors
def calc_norm(grads):
	norm = 0.0
	for g_ in grads:
		if g_ is not None:
			norm += (g_**2).sum()
	return np.sqrt(norm.item())

def get_body_end(model):
	pos = 0
	for k, _ in model.named_parameters():
		if '_text_field_embedder' in k:
			pos += 1
	return pos

# For collating data
def collate(examples, pad_token_id):
	return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)

def add_modelling_options(parser):
		# primary task arguments
	parser.add_argument("--prim-task-id", default=None, type=str, required=True, help='The ID of the primary task')
	parser.add_argument("--dev_data_file", default=None, type=str, required=True, help='The input dev data file (a text file)')
	parser.add_argument("--test_data_file", default=None, type=str, required=True, help='The input test data file (a text file)')
	parser.add_argument(
		"--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
	)
	
	parser.add_argument("--classf_betas", type=str, default="(0.9,0.98)")
	parser.add_argument("--classf_dev_lr", type=float, default=1e-4, help="Learning rate of dev-head")
	parser.add_argument("--classf_dev_wd", type=float, default=0.1)
	parser.add_argument("--classf_ft_iters", type=int, default=10, help='Number of finetuning iterations')
	parser.add_argument("--classf_max_seq_len", type=int, default=512)
	parser.add_argument("--classifier_dropout", type=float, default=0.1)
	parser.add_argument("--classf_iter_batchsz", type=int, default=8, help='Batch Size per iteration. True batch_sz is this x number of grad accumulation steps')
	parser.add_argument('--dev_batch_sz', type=int, default=128, help='Batch sz for dev-set for meta-learning')
	parser.add_argument(
		"--no_final_finetuning",
		action='store_true',
		help='turns off further task-specific finetuing'
	)
	parser.add_argument("--n-runs-classf", type=int, default=1)
	parser.add_argument("--classf_wd", type=float, default=0.1)
	parser.add_argument("--classf_ft_patience", type=int, default=3, help='finetuning patience iterations')
	parser.add_argument("--classf-metric", type=str, default='f1', choices=['f1', 'accuracy'])
	parser.add_argument("--classf_warmup_frac", type=float, default=0.06)
	parser.add_argument(
		"--eval_every",
		type=int,
		default=50,
		help='How frequently to evaluate the model so we can cache the best version'
	)
	parser.add_argument("--classf_patience", type=int, default=5)
	parser.add_argument("--classf_lr", type=float, default=2e-5, help="Learning rate of classifier")
	parser.add_argument("--classf_ft_lr", type=float, default=2e-6, help="Learning rate of classifier for finetuning")

	return parser


class ModelWithLMHead(nn.Module):
	def __init__(self, base_model, model_name):
		super().__init__()
		self._text_field_embedder = base_model
		config = AutoConfig.from_pretrained(model_name)
		self.lm_head = RobertaLMHead(config)
		self._loss = torch.nn.CrossEntropyLoss(reduction='none')
	
	def forward(self, tokens, labels, attn_mask=None):
		embedded_text = self._text_field_embedder(tokens, attention_mask=attn_mask)
		if isinstance(tokens, dict):
			mask = get_text_field_mask(tokens).float()
		else:
			mask = None
			embedded_text = embedded_text[1][-1]
		logits = self.lm_head(embedded_text)
		loss = self._loss(logits.view(-1, logits.shape[-1]), labels.long().view(-1))
		output_dict = {}
		output_dict["loss_full"] = loss
		output_dict["loss"] = (loss.sum() / len(loss.nonzero()))
		return output_dict

class ModelWithAuxTasks(AutoModel):
	''' Wrapper around a basic model so we can do gradient surgery on the model'''
	def __init__(
					self,
					model_name,
					base_model,
					searchOpts,
					args,
					primary_task_info,  # Dictionary of task_id : task_file
					max_seq_len=512,
					dropout=0.0,
					embedding_dim=768,
					ff_multiplier=1,
					num_layers=1,
					max_norm=1.0,
					save_path=None,
					batch_sz=8,
					grad_accum_factor=8,
					dev_batch_sz=128,
	):
		assert save_path is not None, 'Invalid Save Path Provided for Classifier Head'
		assert isinstance(primary_task_info, dict), 'Invalid type of base_task_dataset_files. Expected Dict'
		assert 'prim_task_id' in primary_task_info, 'No primary task id is given'

		self.options = args
		self.save_path = save_path
		self.base_model = base_model
		self.primary_task_info = primary_task_info
		# Todo [ldery] - might need to extend this for multiple primary tasks like GLUE
		self.primary_task_id  = primary_task_info['prim_task_id']
		primary_task_data_dict = {
			'train': primary_task_info['train_fname'],
			'dev': primary_task_info['dev_fname'],
			'test': primary_task_info['test_fname'],
		}
		self.datasets = self.setup_datasets(primary_task_data_dict, model_name, max_seq_len, lazy=False)
		# Cached for later use
		self.embedding_dim = embedding_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.ff_multiplier = ff_multiplier
		# Setting up output heads
		self.model_name = model_name
		self.base_model = base_model
		self.setup_heads(searchOpts, dropout, embedding_dim, num_layers, ff_multiplier)
		self.batch_sz = batch_sz
		self.max_norm = 1.0
		self.max_seq_len = max_seq_len
		self.grad_accum_factor = grad_accum_factor
		# Save grads of auxiliary losses
		self.auxloss_cosines = defaultdict(list)
		self.dev_batch_sz = dev_batch_sz # Batch Size for dev-set
		# Instantiating a summary writer here
		log_dir = os.path.join(args.output_dir, 'TBoardLogger')
		print('This is the tensorboard log dir', log_dir)
		self.tboard_writer = SummaryWriter(log_dir=log_dir)
		self.setup_perf_monitors()

	def setup_perf_monitors(self):
		self.dev_head_perfs = defaultdict(list)
		self.per_param_dp = defaultdict(list)
		self.weight_stats = defaultdict(list)
		self.config_losses_and_weights = defaultdict(list)


	def setup_datasets(self, dataset_split_dict, model_name, max_seq_len, label_vocab=None, lazy=False):
		# Instantiate dataset reader
		datasets = defaultdict(dict)
		tokenizer = PretrainedTransformerTokenizer(model_name, do_lowercase=False, start_tokens=["<s>"], end_tokens=["</s>"])
		indexers = {'tokens': PretrainedTransformerIndexer(model_name, do_lowercase=False)}
		dataset_reader = TextClassificationJsonReaderWithSampling(
							token_indexers=indexers, tokenizer=tokenizer,
							max_sequence_length=max_seq_len, lazy=lazy
						)
		# Read from the dataset
		pretrain_vocab = tokenizer._tokenizer.encoder
		for idx_, fname in dataset_split_dict.items():
			all_samples = dataset_reader._read(fname)
			all_sentences, all_instances = [], []
			lens = []
			for instance in all_samples:
				tokens = instance.fields['tokens']
				tokens.index(pretrain_vocab)
				sentence = tokens.as_tensor(tokens.get_padding_lengths())['tokens']
				all_sentences.append(sentence)
				all_instances.append(instance)
				lens.append(sentence.shape[0])
			print('These are the length statistics of {}'.format(fname))
			print('Mean = {} | Std = {} | max = {}'.format(np.mean(lens), np.std(lens), max(lens)))
			if label_vocab is not None:
				vocab = label_vocab
			else:
				vocab = Vocabulary.from_instances(all_instances)
				label_vocab = vocab

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

	def setup_heads(self, searchOpts, dropout, embedding_dim, num_layers, ff_multiplier):
		self.head_list = []
		# Setup the primary task here
		self.head_list.append(self.primary_task_info['prim_task_id'])
		vocab = self.datasets['train']['vocab']
		self.setup_classifier(dropout, self.primary_task_info['prim_task_id'], vocab, embedding_dim, ff_multiplier, num_layers=num_layers)
		# Setup the other outputs here
		for aux_loss_config in searchOpts.get_valid_configs():
			key_ = ".".join([str(x) for x in aux_loss_config])
			if searchOpts.is_tokenlevel(aux_loss_config[-1]):
				# We are doing token level classification here
				vocab_tokens, is_lm = searchOpts.is_tokenlevel_lm(aux_loss_config[-1])
				if not is_lm:
					vocab = Vocabulary()
					vocab.add_tokens_to_namespace(vocab_tokens)
					self.setup_seq_tagger(dropout, key_, vocab, embedding_dim, ff_multiplier, num_layers=1, num_labels=len(vocab_tokens))
				else:
					this_model = ModelWithLMHead(self.base_model, self.model_name)
					setattr(self, 'AuxHead-{}'.format(key_), this_model)
			elif searchOpts.is_dot_prod(aux_loss_config[-1]):
				# Todo [ldery] - need to implement this
				raise NotImplementedError('Not yet implemented')
			elif searchOpts.is_sent_classf(aux_loss_config[-1]):
				vocab = Vocabulary()
				vocab.add_tokens_to_namespace(searchOpts.get_vocab(aux_loss_config[-1]))
				self.setup_classifier(dropout, key_, vocab, embedding_dim, ff_multiplier, num_layers=1)
			else:
				raise ValueErorr('Invalid aux_loss_config : ', aux_loss_config)
			self.head_list.append(key_)

	def setup_seq_tagger(self, dropout, task_idx, vocab, embedding_dim, ff_multiplier, num_layers=1, num_labels=None):
		text_field_embedder = self.base_model
		hidden_dim = embedding_dim * ff_multiplier
		feedforward = FeedForward(
									embedding_dim, num_layers, hidden_dims=hidden_dim,
									activations=torch.nn.ReLU(), dropout=dropout
								)
		classifier = BasicSequenceTagger(vocab, text_field_embedder, feedforward, embedding_dim, num_labels=num_labels, dropout=dropout, initializer=None)
		classifier.to(self.base_model.device)
		setattr(self, 'AuxHead-{}'.format(task_idx), classifier)
		return classifier

	def setup_classifier(self, dropout, task_idx, vocab, embedding_dim, ff_multiplier, num_layers=1):
		text_field_embedder = self.base_model
		seq2vec_encoder = CLSPooler(embedding_dim)
		hidden_dim = embedding_dim * ff_multiplier
		feedforward = FeedForward(
									embedding_dim, num_layers, hidden_dims=hidden_dim,
									activations=torch.nn.Tanh(), dropout=dropout
								)
		classifier = BasicClassifierWithF1(vocab, text_field_embedder, seq2vec_encoder, feedforward, dropout=dropout, initializer=None)
		classifier.to(self.base_model.device)
		setattr(self, 'AuxHead-{}'.format(task_idx), classifier)
		return classifier

	def get_classifier_params(self, keys=None, withbase=False):
		param_list = []
		# Get all the classifier params if keys is not specified
		if keys is None:
			keys = self.head_list
		for _, key in enumerate(keys):
			this_head = getattr(self, "AuxHead-{}".format(key), None)
			assert this_head is not None, 'Auxiliary Classifier {} not found'.format(key)
			if withbase and key == self.primary_task_info['prim_task_id']:
				param_list.extend(this_head.named_parameters())
			else:
				filtered_param_list = [param for pname, param in this_head.named_parameters() if '_text_field_embedder' not in pname]
				param_list.extend(filtered_param_list)
		return param_list

	def to(self, device):
		self.base_model.to(device)
		for key in self.head_list:
			this_head = getattr(self, "AuxHead-{}".format(key), None)
			assert this_head is not None, 'Auxiliary Head {} not found'.format(key)
			this_head.to(device)
			# Since we have moved this to gpu, we need to re-set the base.
			this_head._text_field_embedder = self.base_model

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
			}
		for key in self.head_list:
			this_head = getattr(self, "AuxHead-{}".format(key), None)
			assert this_head is not None, 'Auxiliary Classifier {} not found'.format(key)
			save_dict[key] = this_head.state_dict()
		torch.save(
			save_dict,
			path
		)

	def set_save_path(self, save_path):
		self.save_path = save_path


	def load(self):
		# We are assuming that what we care about is the primary task parameters
		primary_classifier = getattr(self, "AuxHead-{}".format(self.primary_task_info['prim_task_id']), None)
		assert primary_classifier is not None, 'Cannot find primary task classifier head to load'
		state_dict = torch.load(self.save_path)
		for key in self.head_list:
			this_head = getattr(self, "AuxHead-{}".format(key), None)
			assert this_head is not None, 'Auxiliary Head {} not found'.format(key)
			this_head.load_state_dict(state_dict[key])

		if hasattr(self, 'optimizer') and ('optimizer_sd' in state_dict):
			self.optimizer.load_state_dict(state_dict['optimizer_sd'])
			self.ft_lr_scheduler = state_dict['scheduler']

		self.base_model = this_head._text_field_embedder

	def load_primary(self, device):
		# We are assuming that what we care about is the primary task parameters
		primary_head = getattr(self, "AuxHead-{}".format(self.primary_task_info['prim_task_id']), None)
		assert primary_head is not None, 'Cannot find primary task classifier head to load'
		state_dict = torch.load(self.save_path)
		primary_head.load_state_dict(state_dict[self.primary_task_info['prim_task_id']])
		primary_head.to(device)
		self.base_model = primary_head._text_field_embedder

	# Evaluate the classifier
	def evaluate_classifier(self, set_='dev'):
		assert set_ in ['dev', 'test'], 'Wrong dataset specified'
		dataset = self.datasets[set_]
		prim_head = getattr(self, "AuxHead-{}".format(self.primary_task_info['prim_task_id']), None)
		assert prim_head is not None, 'Auxiliary Classifier {} not found'.format(key)
		# Run the classifier
		torch.cuda.empty_cache()
		# reset the metrics before running new stuff
		try:
			_ = self.get_metrics(head_=prim_head, reset=True)
		except:
			print('This classifier does not need to reset metrics.')
		prim_head.eval()
		with torch.no_grad():
			for samples in self.dataset_iterator(dataset, batchsz=self.batch_sz):
				_ = prim_head(*samples)
		prim_head.train()
		# Get the metrics from the classifier
		torch.cuda.empty_cache()
		return self.get_metrics(head_=prim_head, reset=True)

	# Get metrics from a particular classifier. Defaults to the primary task classifier
	def get_metrics(self, head_=None, reset=False):
		if head_ is None:
			head_ = getattr(self, "AuxHead-{}".format(self.primary_task_info['prim_task_id']), None)
			assert head_ is not None, 'Auxiliary Head {} not found'.format(key)
			# Get the metrics from the classifier
		return head_.get_metrics(reset=reset)

	# Get samples for a task
	def get_classifier_samples(self, dataset, nsamples):
		num_egs = len(dataset['tokens'])
		idxs = np.random.choice(num_egs, size=nsamples, replace=False)
		sentences, labels = [dataset['tokens'][i] for i in idxs], dataset['labels'][idxs]
		sentences = collate(sentences, dataset['pad_idx'])
		sentences = sentences.to(self.base_model.device)
		labels = torch.IntTensor(labels).to(sentences.device)
		return sentences, labels

	# This function should set the dev head.
	# This function should also return the gradients w.r.t the dev-head
	def set_dev_head(self):
		# Learn the head here
		dev_head = self.learn_dev_head()
		# Get the dev gradient here
		dev_dataset = self.datasets['dev']
		dev_sent, dev_labels = self.get_classifier_samples(dev_dataset, self.batch_sz)
		try:
			loss_ = dev_head(dev_sent, dev_labels)['loss']
			gradients = torch.autograd.grad(loss_, dev_head.parameters(), allow_unused=True)
		except RuntimeError as e:
			if 'out of memory' in str(e):
				print('| WARNING: ran out of memory, retrying batch in set_dev_head')
				torch.cuda.empty_cache()
				loss_ = dev_head(dev_sent, dev_labels)['loss']
				gradients = torch.autograd.grad(loss_, this_head.parameters(), allow_unused=True)
			else:
				raise e
		return gradients

	# This function resets the dev-head. We use this anytime we need a new approximation of the dev head
	def reset_dev_head(self):
		dev_head_name = "AuxHead-{}-{}".format('dev', self.primary_task_info['prim_task_id'])
		this_head = getattr(self, dev_head_name, None)
		if this_head is not None:
			del this_head
		else:
			# This means that we had to re-use the previous dev-head from past estimation because we couldn't
			# re-estimate it.
			print('The dev head should be not be none. Probably because of the oom')
		setattr(self, dev_head_name, None)

	# This function learns the dev_head
	def learn_dev_head(self):
		assert hasattr(self, 'options'), 'The options need to be set for training of the dev head'
		head_name =  "{}-{}".format('dev', self.primary_task_info['prim_task_id'])
		this_head = getattr(self, "AuxHead-{}".format(head_name), None)
		dev_params = None
		if this_head is None:
			# Need to instantiate the classifier head
			this_head = self.setup_classifier(
								self.dropout, head_name, self.datasets['dev']['vocab'],
								self.embedding_dim, self.ff_multiplier, num_layers=self.num_layers
							)
			# Setup optimizer for dev head
			dev_params = self.get_classifier_params([head_name], withbase=False)
			assert dev_params is not None, 'Dev Params should have been instantiated above'
			dev_optim =  AdamW(
									dev_params, betas=eval(self.options.classf_betas),
									weight_decay=self.options.classf_dev_wd, lr=self.options.classf_dev_lr
								)
		else:
			return this_head # We train this once and re-use

		# This is the first time instantiating this head so we need to train it
		assert dev_optim is not None, 'The optimizer for the dev head has not been instantiated'

		# perform gradient descent to get the dev-head
		# Todo [ldery] - restore this to 'dev' if it turns out this is not the issue
		samples = self.get_classifier_samples(self.datasets['train'], self.dev_batch_sz)
		prev_loss_, tol = 1e10, 1e-3
		all_metrics = [[], [], []]
		for i in range(self.options.classf_ft_iters):
			output = this_head(*samples)
			loss_ = output['loss']
			# This ensures that we only train the dev-head and keep the body fixed
			grads = torch.autograd.grad(loss_, dev_params, allow_unused=True)
			for p, g in zip(dev_params, grads):
				assert g is not None, 'This should have a gradient'
				p.grad = g
			dev_optim.step()

			# Save performance for analysis
			metrics = self.get_metrics(head_=this_head, reset=True)
			all_metrics[0].append(metrics['f1'])
			all_metrics[1].append(metrics['accuracy'])
			all_metrics[2].append(loss_.item())

			if abs(loss_ - prev_loss_) < tol:
				break
			prev_loss_ = loss_.item()
			del grads
			del loss_
			torch.cuda.empty_cache()

		# Save performance for analysis
		self.dev_head_perfs['f1'].append(np.mean(all_metrics[0]))
		self.dev_head_perfs['accuracy'].append(np.mean(all_metrics[1]))
		self.dev_head_perfs['loss'].append(np.mean(all_metrics[2]))
		return this_head

	def close_writer(self):
		self.tboard_writer.close()

	def push_to_tensorboard(self, step_):
		# push the config losses and their weights
		for k, v in self.dev_head_perfs.items():
			self.tboard_writer.add_scalar('dev.head.perfs.{}'.format(k), v[-1], step_)

		aux_cosines, losses_, weights_, prods_  = {}, {}, {}, {}
		norms_ = {}
		for k, v in self.weight_stats.items():
			values = [x[-1] for x in v[-self.grad_accum_factor:]]
			aux_cosines[k] = np.mean(values)

			values = [x[1] for x in v[-self.grad_accum_factor:]]
			norms_[k] = np.mean(values)

			v_l = self.config_losses_and_weights[k]
			values = [x[0] for x in v_l[-self.grad_accum_factor:]]
			losses_[k] = np.mean(values)
			
			values = [x[1] for x in v_l[-self.grad_accum_factor:]]
			weights_[k] = np.mean(values)
			prods_[k] = weights_[k] * losses_[k]

		self.tboard_writer.add_scalars('aux.cosines', aux_cosines, step_)
		self.tboard_writer.add_scalars('aux.losses', losses_, step_)
		self.tboard_writer.add_scalars('aux.weights', weights_, step_)
		self.tboard_writer.add_scalars('aux.lossxweights', prods_, step_)
		self.tboard_writer.add_scalars('aux.gradnorms', norms_, step_)

	def push_metric_to_tensorboard(self, metric, step_, metric_name):
		self.tboard_writer.add_scalars(metric_name, metric, step_)

	# At the end of this, all the model gradients should be populated appropriately 
	def get_grads_with_auxiliaries(self, aux_config_w_batch, searchOpts):
		this_head = None
		if not hasattr(self, 'body_params_end'):
			this_head = getattr(self, "AuxHead-{}".format(self.primary_task_info['prim_task_id']), None)
			assert this_head is not None, 'Cannot find primary task head'
			self.body_params_end = get_body_end(this_head)
		dev_id = "dev-{}".format(self.primary_task_info['prim_task_id'])
		dev_head_grads = self.set_dev_head()[:self.body_params_end]
		dev_norm = calc_norm(dev_head_grads)
		auxloss_cosines = {}

		key_order = [self.primary_task_info['prim_task_id']]
		key_order.extend(aux_config_w_batch.keys())
		sent_dict, labels = self.get_classifier_samples(self.datasets['train'], self.batch_sz)
		prim_batch = {'input':sent_dict , 'output':labels, 'rep_mask': None}
		aux_config_w_batch.update({self.primary_task_info['prim_task_id']: prim_batch})
		aux_weights, prim_weight = searchOpts.get_weighttensor_nograd()
		
		prim_norm = None
		for aux_loss_config in key_order:
			batch = aux_config_w_batch[aux_loss_config]
			is_prim = not isinstance(aux_loss_config, tuple)
			human_readable = searchOpts.get_config_human_readable(aux_loss_config) if not is_prim else aux_loss_config
			if not is_prim:
				task_id = ".".join([str(x) for x in aux_loss_config])
			else:
				task_id = aux_loss_config
			this_head = getattr(self, "AuxHead-{}".format(task_id), None)
			assert this_head is not None, 'Auxiliary Classifier {} not found'.format(task_id)
			for k, v in batch.items():
				if isinstance(v, torch.Tensor):
					batch[k] = v.cuda()
			input_, output_, rep_mask = batch['input'], batch['output'], batch['rep_mask']
			loss_ = this_head(input_, output_, attn_mask=rep_mask)['loss']
			gradients = torch.autograd.grad(loss_, this_head.parameters(), allow_unused=True, retain_graph=True)

			this_grads = gradients[:self.body_params_end]
			task_norm = calc_norm(this_grads)
			per_param_dp = []
			cos_sim = dot_prod(dev_head_grads, this_grads, ppdp=per_param_dp)
			cos_sim = (cos_sim / (dev_norm * task_norm)) 
			self.weight_stats[human_readable].append((dev_norm.item(), task_norm.item(), cos_sim))
			cos_sim = cos_sim / self.grad_accum_factor
			self.per_param_dp[human_readable].append(per_param_dp)
			searchOpts.update_grad(aux_loss_config, -cos_sim)
			# Add this to the model gradients
			if not is_prim:
				this_weight = aux_weights[aux_loss_config[0], aux_loss_config[1], aux_loss_config[2], aux_loss_config[3]].item()
			else:
				this_weight = prim_weight.item()
			if prim_norm is None:
				assert is_prim, 'The primary task should be the first to be run'
				prim_norm = task_norm
			norm_weighting = 1.0 # prim_norm / task_norm
			weighted_loss = (loss_ * this_weight * norm_weighting) / self.grad_accum_factor
			weighted_loss.backward()
			self.config_losses_and_weights[human_readable].append((loss_.item(), this_weight))


	# We train the primary head. This is further finetuning on top pre-training
	def train_primary(self, n_iters, optimizer, lr_scheduler, max_grad_norm, patience=3, metric='f1'):
		# Setup Optimizer and stuff
		best_iter = 0
		assert self.datasets is not None, 'Need to instantiate the dataset'
		dataset = self.datasets['train']
		key = self.primary_task_info['prim_task_id']
		prim_head = getattr(self, "AuxHead-{}".format(key), None)
		assert prim_head is not None, 'Auxiliary Classifier {} not found'.format(key)
		prim_head.train()
		self.perfs = defaultdict(list)
		iters_since_improvement = 0
		for iter_ in range(n_iters):
			print('Currently on Classifier Epoch {}/{}'.format(iter_ + 1, n_iters))
			iterator = self.dataset_iterator(dataset, shuffle=True)
			total_iters = math.ceil(len(dataset['tokens']) / self.batch_sz)
			# Get the primary classifier
			iterator = tqdm(iterator, total= total_iters, desc="Classifier Train Iterator")
			for idx, samples in enumerate(iterator):
				if (idx + 1) % self.grad_accum_factor == 0:
					# We want to take a gradient step
					torch.nn.utils.clip_grad_norm_(prim_head.parameters(), max_grad_norm)
					optimizer.step()
					if lr_scheduler is not None:
						lr_scheduler.step()
					optimizer.zero_grad()
				output_dict = prim_head(*samples)
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
			metric_idx = 0 if metric == 'f1' else 1
			if dev_metrics[metric] >= self.perfs['dev'][best_iter][metric_idx]:
				best_iter = iter_
				iters_since_improvement = 0
			else:
				iters_since_improvement += 1
				if iters_since_improvement >= patience:
					print('Breaking because we have no improvement in {} epochs'.format(patience))
					break
		best_f1, best_acc = self.perfs['test'][best_iter]
		return best_f1, best_acc, self.perfs, self.perfs['dev'][best_iter]


	def dataset_iterator(self, dataset, shuffle=False, batchsz=-1):
		if batchsz < 0:
			batchsz = self.batch_sz
		total_egs = len(dataset['tokens'])
		num_batches = math.ceil(total_egs / batchsz)
		if shuffle:
			idxs = np.random.permutation(total_egs)
		else:
			idxs = list(range(total_egs))
		for i in range(num_batches):
			this_idxs = idxs[(i * batchsz): ((i + 1) * batchsz)]
			sentences = [dataset['tokens'][id_] for id_ in this_idxs]
			labels = dataset['labels'][this_idxs]
			sentences = collate(sentences, dataset['pad_idx'])
			sentences = sentences.to(self.base_model.device)
			labels = torch.IntTensor(labels).to(self.base_model.device)
			yield sentences, labels
			del sentences
			del labels
	
	def get_prim_dataset_len(self, batchsz=-1):
		if batchsz < 0:
			batchsz = self.batch_sz
		total_egs = len(self.datasets['train']['tokens'])
		num_batches = math.ceil(total_egs / batchsz)
		return num_batches

	def forward(*args, **kwargs):
		raise NotImplementedError(
						'Forward Method Should not be called directly on'
						' this class. Use get_grads_with_auxiliaries instead'
				)

