### THIS FILE IS COPIED FROM THE HUGGINGFACE REPOSITORY FOR CONVENIENCE.

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import pdb
from typing import Dict, List, Tuple
import gc

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from .endtask_aware import ModelWithAuxTasks
from .endtask_auxtasks import get_auxtask_files
from transformers import (
	MODEL_WITH_LM_HEAD_MAPPING,
	WEIGHTS_NAME,
	AdamW,
	AutoConfig,
	AutoModelWithLMHead,
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
	get_linear_schedule_with_warmup,
)


try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TextDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
		assert os.path.isfile(file_path)

		block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

		directory, filename = os.path.split(file_path)
		cached_features_file = os.path.join(
			directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
		)

		if os.path.exists(cached_features_file) and not args.overwrite_cache:
			logger.info("Loading features from cached file %s", cached_features_file)
			with open(cached_features_file, "rb") as handle:
				self.examples = pickle.load(handle)
		else:
			logger.info("Creating features from dataset file at %s", directory)

			self.examples = []
			with open(file_path, encoding="utf-8") as f:
				text = f.read()

			tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

			for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
				self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
			# Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
			# If your dataset is small, first you should loook for a bigger one :-) and second you
			# can change this behavior by adding (model specific) padding.

			logger.info("Saving features into cached file %s", cached_features_file)
			with open(cached_features_file, "wb") as handle:
				pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		return torch.tensor(self.examples[item], dtype=torch.long)

def get_tokenized_file(file_path:str, tokenizer: PreTrainedTokenizer, block_size=512, lazy=False):
	logger.info("Creating features from dataset file at %s", file_path)
	logger.info("Reading Line by Line")
	lines = []
	with open(file_path, encoding="utf-8") as f:
		for line in f.readlines():
			if len(line) > 0 and not line.isspace():
				lines.append(line)
	logger.info("Done Reading Line By Line. About to pass through the tokenize")
	if lazy:
		return lines
	return tokenizer.batch_encode_plus(lines, truncation=True, add_special_tokens=True, max_length=block_size)["input_ids"]


class LineByLineTextDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, args, lazy:bool, file_path: str, block_size=512):
		assert os.path.isfile(file_path)
		# Here, we do not cache the features, operating under the assumption
		# that we will soon use fast multithreaded tokenizers from the
		# `tokenizers` repo everywhere =)
		self.lazy = lazy
		self.block_size = block_size
		# use the lazy option when the dataset is too big to reasonably try to fit in memory as tensors
		if lazy:
			self.tokenizer = tokenizer
		self.examples = get_tokenized_file(file_path, tokenizer, block_size, lazy=lazy)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		tokenized = self.examples[i]
		if self.lazy:
			tokenized = self.tokenizer.encode_plus(tokenized, truncation=True, add_special_tokens=True, max_length=self.block_size)["input_ids"]
		return torch.tensor(tokenized, dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
	file_paths = args.eval_data_file if evaluate else args.train_data_file
	assert len(args.train_data_file) == len(args.aux_task_names), 'Mismatch between the number of train files for MLM and the number of aux task names'
	datasets = {}
	for idx, file_path in enumerate(file_paths):
		task_name = args.aux_task_names[idx]
		if args.line_by_line:
			lazy = (not evaluate) and args.lazy_dataset
			datasets[task_name] = LineByLineTextDataset(tokenizer, args, lazy=lazy, file_path=file_path, block_size=args.block_size)
		else:
			datasets[task_name] = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
	return datasets


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
	ordering_and_checkpoint_path = []
	glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
	for path in glob_checkpoints:
		if use_mtime:
			ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
		else:
			regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
			if regex_match and regex_match.groups():
				ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

	checkpoints_sorted = sorted(ordering_and_checkpoint_path)
	checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
	return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
	if not args.save_total_limit:
		return
	if args.save_total_limit <= 0:
		return

	# Check if we should delete older checkpoint(s)
	checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
	if len(checkpoints_sorted) <= args.save_total_limit:
		return

	number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
	checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
	for checkpoint in checkpoints_to_be_deleted:
		logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
		shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
	""" Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

	if tokenizer.mask_token is None:
		raise ValueError(
			"This tokenizer does not have a mask token which is necessary for masked"
			" language modeling. Remove the --mlm flag if you want to use this tokenizer."
		)

	labels = inputs.clone()
	# We sample a few tokens in each sequence for masked-LM training
	# (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
	probability_matrix = torch.full(labels.shape, args.mlm_probability)
	special_tokens_mask = [
		tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
	]
	probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
	if tokenizer._pad_token is not None:
		padding_mask = labels.eq(tokenizer.pad_token_id)
		probability_matrix.masked_fill_(padding_mask, value=0.0)
	masked_indices = torch.bernoulli(probability_matrix).bool()
	labels[~masked_indices] = -100  # We only compute loss on masked tokens

	# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
	indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
	inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

	# 10% of the time, we replace masked input tokens with random word
	indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
	random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
	inputs[indices_random] = random_words[indices_random]

	# The rest of the time (10% of the time) we keep the masked input tokens unchanged
	return inputs, labels

def save_chkpt(args, id_, model, tokenizer, optimizer, scheduler, rotate_chkpt=True):
	checkpoint_prefix = "checkpoint"
	output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, id_))
	os.makedirs(output_dir, exist_ok=True)
	model_to_save = (
		model.module if hasattr(model, "module") else model
	)  # Take care of distributed/parallel training
	model_to_save.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)

	torch.save(args, os.path.join(output_dir, "training_args.bin"))
	logger.info("Saving model checkpoint to %s", output_dir)

	if rotate_chkpt:
		_rotate_checkpoints(args, checkpoint_prefix)

	torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
	torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
	logger.info("Saving optimizer and scheduler states to %s", output_dir)

# Run a batch of data through the model whilst checking for out of memory errors
def run_batch(model, batch, tokenizer, args, task_name, try_again=False):
	try :
		inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)
		model.train()
		outputs = model(inputs, labels=labels, head_name=task_name)
	except RuntimeError as e:
		pdb.set_trace()
		gc.collect()
		if 'out of memory' in str(e):
			if try_again:
				print('| WARNING: ran out of memory during forward. Trying batch again')
			else:
				print('| WARNING: ran out of memory during forward. Skipping batch')
		else:
			print('Run into this new error : ', str(e))
		torch.cuda.empty_cache()
		if not try_again:
			return None
		else:
			outputs = run_batch(model, batch, tokenizer, args, task_name, try_again=False)
	return outputs

# Process a batch of data for a particular task
def process_task_batch(auxTaskModel, model, batch, tokenizer, args, task_name, sample_prim_grad=False):
	outputs = run_batch(model, batch, tokenizer, args, task_name)

	# This returns none if we aren't able to process the batch even after clearing
	# the cuda cache after an out of memory erorr and re-trying
	loss_ = 0
	if outputs is not None:
		loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
		# Store the gradients for the meta-learner
		if args.n_gpu > 1:
			loss = loss.mean()  # mean() to average on multi-gpu parallel training
		if auxTaskModel.alpha_generator_algo.is_meta:
			gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
			auxTaskModel.set_mlm_grads(gradients, aux_task_name=task_name)
		scale = 0 if args.no_mlm_weight else auxTaskModel.alpha_generator_algo[task_name]
		loss = loss * scale
		grad_accum_factor = 1
		if args.gradient_accumulation_steps > 1:
			loss = loss / args.gradient_accumulation_steps
			grad_accum_factor = args.gradient_accumulation_steps

		if args.fp16:
			with amp.scale_loss(loss, optimizer) as scaled_loss:
				scaled_loss.backward()
		else:
			reached_sample_grad = False
			try:
				if not auxTaskModel.alpha_generator_algo.is_meta:
					loss.backward()
				else:
					assert gradients is not None, ' Gradients should be set by now'
					scale = auxTaskModel.alpha_generator_algo[task_name] / args.gradient_accumulation_steps
					with torch.no_grad():
						for (p, g) in zip(model.parameters(), gradients):
							if g is None:
								continue
							if p.grad is None:
								p.grad = torch.zeros_like(p)
							p.grad.add_(g * scale)
							del g
					torch.cuda.empty_cache()

				loss_ = loss.item()
				if sample_prim_grad: # Note that we must call this after all the auxiliary tasks have been run
					reached_sample_grad = True
					auxTaskModel.classifier_sample_grad()
			except RuntimeError as e:
				if 'out of memory' in str(e) and not reached_sample_grad:
					print('| WARNING: ran out of memory during backward. Skipping Batch')
				if 'out of memory' in str(e):
					print('| WARNING: ran out of memory during sample grad. Skipping Batch')
				else:
					print('| WARNING: crashed but not due to memory error')
				gc.collect()
				torch.cuda.empty_cache()
	return loss_


def train(
			args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
			auxTaskModel: ModelWithAuxTasks = None) -> Tuple[int, float]:
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	if args.n_gpu > 1:
		# Unevenly split the batches among the gpu. Since gpu(0) is where the gradient accumulation happens
		# we put less examples on this gpu so as to prevent out of memory error
		this_chunks = [args.per_gpu_train_batch_size for i in range(torch.cuda.device_count())]
		this_chunks[0] = args.base_batchsz
		setattr(args, 'batching_chunks', this_chunks)
		setattr(args, 'train_batch_size', sum(this_chunks))
	else:
		args.train_batch_size = args.per_gpu_train_batch_size

	def collate(examples: List[torch.Tensor]):
		# Try to balance the examples across the gpus
		lens = [len(x) for x in examples]
		order = np.argsort(lens)
		new_order, order = (order[:args.base_batchsz]).tolist(), order[args.base_batchsz:]
		for idx_ in range(args.n_gpu):
			new_order.extend(order[idx_::args.n_gpu])
		examples = [examples[x] for x in new_order]
		if tokenizer._pad_token is None:
			return pad_sequence(examples, batch_first=True)
		return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

	train_dataloader = {}
	max_dataset_len, largest_dataset_name = -1, None
	for task_name, dataset in train_dataset.items():
		train_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
		bsz_ = args.train_batch_size #args.classf_iter_batchsz if 'TAPT' in task_name else args.train_batch_size
		train_dataloader[task_name] = DataLoader(
			dataset, sampler=train_sampler, batch_size=bsz_, collate_fn=collate, drop_last=True
		)
		if max_dataset_len < len(dataset):
			max_dataset_len = len(dataset)
			largest_dataset_name = task_name

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (max_dataset_len // args.gradient_accumulation_steps) + 1
		logger.info('The number of epochs is : {}'.format(args.num_train_epochs))
	else:
		t_total = max_dataset_len // args.gradient_accumulation_steps * args.num_train_epochs

	model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
	model.resize_token_embeddings(len(tokenizer))

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	# Begin change by [ldery]
	# Setup the optimizer for the base model
	optimizer = AdamW(
						optimizer_grouped_parameters, betas=eval(args.classf_betas),
						lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.base_wd
					)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=int(args.classf_warmup_frac * t_total), num_training_steps=t_total
	)

	# Setup an optimizer for the classifier
	classifier_params = auxTaskModel.get_classifier_params(withbase=False)
	classifier_optim = AdamW(
								classifier_params, betas=eval(args.classf_betas),
								weight_decay=args.classf_wd, lr=args.classf_lr
							)
	classifier_scheduler = get_linear_schedule_with_warmup(
		classifier_optim, num_warmup_steps=int(args.classf_warmup_frac * t_total), num_training_steps=t_total
	)
	# Setup the auxiliary task weight generator
	args.prim_start = int(args.prim_start * t_total)
	args.train_epochs = t_total
	args.alt_freq = int(args.alt_freq * t_total)
	print('This is the total iters {}. This is the frequency {}'.format(t_total, args.alt_freq))
	args.alt_freq = max(args.alt_freq, 2) # Setting a minimum frequency of 2
	auxTaskModel.setup_alpha_generator(args)
	# End change by [ldery]

	# Check if saved optimizer or scheduler states exist
	if (
		args.model_name_or_path
		and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
		and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
	):
		# Load in optimizer and scheduler states
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	setattr(args, 'model_is_parallel', False)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model, chunks=args.batching_chunks)
		args.model_is_parallel = True
		if args.parallelize_classifiers:
			auxTaskModel.parallelize_classifiers()
	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
		)

	# Setup an optimizer for the grad surgery model classifier

	# Train!
	logger.info("***** Running training *****")
	for k, v in train_dataset.items():
		logger.info(" Task= {} Num examples = {}".format(k, len(v)))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps
		* (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
	)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = {}. Will eval every {}".format(t_total, args.eval_every))

	global_step = 0
	epochs_trained = 0
	steps_trained_in_current_epoch = 0
	# Check if continuing training from a checkpoint
	if args.model_name_or_path and os.path.exists(args.model_name_or_path):
		try:
			# set global_step to gobal_step of last saved checkpoint from model path
			checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
			global_step = int(checkpoint_suffix)
			epochs_trained = global_step // (max_dataset_len // args.gradient_accumulation_steps)
			steps_trained_in_current_epoch = global_step % (max_dataset_len // args.gradient_accumulation_steps)

			logger.info("  Continuing training from checkpoint, will skip to saved global_step")
			logger.info("  Continuing training from epoch %d", epochs_trained)
			logger.info("  Continuing training from global step %d", global_step)
			logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
		except ValueError:
			logger.info("  Starting fine-tuning.")

	tr_loss, logging_loss = 0.0, 0.0

	model.zero_grad()
	train_iterator = trange(
		epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
	)
	set_seed(args)  # Added here for reproducibility
	classifier_dev_perfs = []
	auxTaskModel.alpha_generator_algo.prep_epoch_start(global_step)
	early_stop = False
	for epoch in train_iterator:
		gc.collect()
		if early_stop:
			break
		weights = {k: auxTaskModel.alpha_generator_algo[k] for k in auxTaskModel.alpha_generator_algo.weights.keys()}
		print('\nGStep = {} Weights : '.format(global_step), weights, '\n')
		epoch_iterator = tqdm(train_dataloader[largest_dataset_name], desc="Iteration", disable=args.local_rank not in [-1, 0])\
		
		# Setup Iterators for the other tasks
		aux_task_iterators = {}
		for task_id, task_data in train_dataloader.items():
			if task_id == largest_dataset_name:
				continue
			aux_task_iterators[task_id] = iter(task_data)

		if args.local_rank != -1:
			train_sampler.set_epoch(epoch)

		for step, batch in enumerate(epoch_iterator):
			# Skip past any already trained steps if resuming training
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue

			# there's a better way to write this but feeling lazy
			this_loss = process_task_batch(auxTaskModel, model, batch, tokenizer, args, largest_dataset_name, sample_prim_grad=False)
			tr_loss += this_loss / len(args.aux_task_names)
			other_tasks = list(set(args.aux_task_names) - set([largest_dataset_name]))
			for task_id, task_name in enumerate(other_tasks):
				other_batch = next(aux_task_iterators[task_name], None)
				if other_batch is None:
					aux_task_iterators[task_name] = iter(train_dataloader[task_name])
					other_batch = next(aux_task_iterators[task_name], None)
				assert other_batch is not None, 'We should have more data for {} since we have reset the iterator'.format(task_name)
				is_last_task = task_id == (len(other_tasks) - 1)
				this_loss = process_task_batch(auxTaskModel, model, other_batch, tokenizer, args, task_name, sample_prim_grad=is_last_task)
				tr_loss += this_loss / len(args.aux_task_names) 

			# Todo (ldery) - run batches of the different through the model and do all the necessary gradient computation
			if auxTaskModel.alpha_generator_algo.is_meta:
				# Zero-out the MLM grads because we are done with them at the moment
				for task_name in args.aux_task_names:
					try:
						auxTaskModel.set_mlm_grads(None, aux_task_name=task_name)
					except:
						gc.collect()
						torch.cuda.empty_cache()
						print('Run into error here. There seems to be 2 consecutive memory erors. Skipping')

			if (step + 1) % args.gradient_accumulation_steps == 0:
				if args.fp16:
					torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				# Todo [ldery] - make sure all the gradient stuff you expect to happen is happening
				# Begin change by [ldery]
				torch.nn.utils.clip_grad_norm_(auxTaskModel.get_classifier_params(), args.max_grad_norm)
				classifier_optim.step()
				classifier_scheduler.step()
				classifier_optim.zero_grad()
				# End change by [ldery]
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1
				auxTaskModel.alpha_generator_algo.prep_epoch_start(global_step)
				if auxTaskModel.alpha_generator_algo.is_meta:
					# Reset the dev head and updated the meta-weights
					auxTaskModel.reset_dev_head()
					auxTaskModel.alpha_generator_algo.update_meta_weights()
				torch.cuda.empty_cache()
				# Irregularly report the end of the epoch to save having to do the 
				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					if (
						args.local_rank == -1 and args.evaluate_during_training
					):  # Only evaluate when single GPU otherwise metrics may not average well
						results = evaluate(args, model, tokenizer)
						for key, value in results.items():
							tb_writer.add_scalar("eval_{}".format(key), value, global_step)
					tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
					tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
					logging_loss = tr_loss

				if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
					
					save_chkpt(args, str(global_step), model, tokenizer, optimizer, scheduler, rotate_chkpt=True)
				
				if global_step % (args.eval_every // 2) == 0:
					weights = {k: auxTaskModel.alpha_generator_algo[k] for k in auxTaskModel.alpha_generator_algo.weights.keys()}
					print('\nGStep = {} Weights : '.format(global_step), weights, '\n')

				if global_step % args.eval_every == 0:
					train_metrics = auxTaskModel.get_metrics(reset=True)
					dev_metrics = auxTaskModel.evaluate_classifier(set_='dev')
					test_metrics = auxTaskModel.evaluate_classifier(set_='test')
					for k, v in train_metrics.items():
						print_out = "[{}] | Train : {:.3f} | Dev Set : {:.3f} | Test Set : {:.3f}".format(k, v, dev_metrics[k], test_metrics[k])
						logger.info(print_out)
					classifier_dev_perfs.append(dev_metrics[args.classf_metric])
					if dev_metrics[args.classf_metric] >= max(classifier_dev_perfs):
						# We want to save the best model here
						print('Current best dev f1 = {} achieved. Saving model'.format(dev_metrics[args.classf_metric]))
						logger.info('Now Saving the Classifier Model')
						auxTaskModel.save()
						logger.info('Saving Base Model')
						save_chkpt(args, 'best', model, tokenizer, optimizer, scheduler, rotate_chkpt=False)
					# Record the metrics for the alpha generator
					auxTaskModel.alpha_generator_algo.record_epoch_end(global_step, dev_metrics[args.classf_metric], test_metrics[args.classf_metric])
					if len(classifier_dev_perfs) > args.classf_patience:
						max_ = max(classifier_dev_perfs)
						recent_max = max(classifier_dev_perfs[-args.classf_patience:])
						if recent_max < max_:
							print('Stopping Early at Epoch {} because No Improvement in Dev Set Accuracy'.format(epoch))
							train_iterator.close()
							early_stop = True
							break

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break
	if args.local_rank in [-1, 0]:
		tb_writer.close()
	return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	eval_output_dir = args.output_dir

	eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
	# Just use the first dataset for evaluation - doesn't really matter just doing this so things don't crash.
	eval_dataset = eval_dataset[list(eval_dataset.keys())[0]]

	if args.local_rank in [-1, 0]:
		os.makedirs(eval_output_dir, exist_ok=True)

	if args.n_gpu > 1:
		assert hasattr(args, 'batching_chunks'), 'Batching Chunks is supposed to be set already'
		args.eval_batch_size = sum(args.batching_chunks)
	else:
		args.eval_batch_size = args.per_gpu_eval_batch_size
	# Note that DistributedSampler samples randomly

	def collate(examples: List[torch.Tensor]):
		if tokenizer._pad_token is None:
			return pad_sequence(examples, batch_first=True)
		return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(
		eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
	)

	# multi-gpu evaluate
	if args.n_gpu > 1 and not args.model_is_parallel:
		model = torch.nn.DataParallel(model)

	# Eval!
	logger.info("***** Running evaluation {} *****".format(prefix))
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)
	eval_loss = 0.0
	nb_eval_steps = 0
	model.eval()

	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
		if labels.shape[0] != args.eval_batch_size:
			continue
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)

		with torch.no_grad():
			outputs = model(inputs, labels=labels)
			lm_loss = outputs[0]
			eval_loss += lm_loss.mean().item()
		nb_eval_steps += 1

	eval_loss = eval_loss / nb_eval_steps
	perplexity = torch.exp(torch.tensor(eval_loss))

	result = {"perplexity": perplexity}

	output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
	with open(output_eval_file, "w") as writer:
		logger.info("***** Eval results {} *****".format(prefix))
		for key in sorted(result.keys()):
			logger.info("  %s = %s", key, str(result[key]))
			writer.write("%s = %s\n" % (key, str(result[key])))

	return result


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--train_data_file", nargs='+', default=None, type=str, required=True, help="The input training data file (a text file)."
	)
	parser.add_argument(
		"--aux-task-names", nargs='+', default=None, type=str, help="The names of the auxiliary tasks"
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
		help="The output directory where the model predictions and checkpoints will be written.",
	)
	parser.add_argument(
		"--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
	)

	# Other parameters
	parser.add_argument(
		"--eval_data_file",
		default=None,
		nargs='+',
		type=str,
		help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
	)
	parser.add_argument(
		"--line_by_line",
		action="store_true",
		help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
	)
	parser.add_argument(
		"--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
	)

	parser.add_argument(
		"--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
	)
	parser.add_argument(
		"--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
	)

	parser.add_argument(
		"--config_name",
		default=None,
		type=str,
		help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
	)
	parser.add_argument(
		"--tokenizer_name",
		default=None,
		type=str,
		help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
	)
	parser.add_argument(
		"--cache_dir",
		default=None,
		type=str,
		help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
	)
	parser.add_argument(
		"--block_size",
		default=-1,
		type=int,
		help="Optional input sequence length after tokenization."
		"The training dataset will be truncated in block of this size for training."
		"Default to the model max input length for single sentence inputs (take into account special tokens).",
	)
	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
	)

	parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
	)
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument(
		"--max_steps",
		default=-1,
		type=int,
		help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
	)
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

	parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
	parser.add_argument(
		"--save_total_limit",
		type=int,
		default=None,
		help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
	)
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
	)
	parser.add_argument(
		"--fp16_opt_level",
		type=str,
		default="O1",
		help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
		"See details at https://nvidia.github.io/apex/amp.html",
	)
	parser.add_argument(
		"--base_task_dataset_file",
		type=str,
		help='Name of file for master task'
	)
	
	parser.add_argument(
		"--lazy-dataset",
		action='store_true',
	)
	
	parser.add_argument(
		"--parallelize_classifiers",
		action='store_true'
	)
				
	parser.add_argument(
		"--eval_every",
		type=int,
		default=50
	)
	
	parser.add_argument(
		"--base_batchsz",
		type=int,
		default=6
	)
	
	parser.add_argument(
		"--no_final_finetuning",
		action='store_true',
		help='turns off further task-specific finetuing'
	)

	# Begin Change [ldery]
	# Weighting Algorithm Specifics
	parser.add_argument("--alpha_update_algo", type=str, default='default', choices=['default', 'alt', 'warm_up_down', 'phase_in', 'meta'])
	parser.add_argument("--meta_learn_aux", action='store_true', help='Used to activate a meta-learning algo')
	parser.add_argument('--alt-freq', type=float, default=0.1, help='If using alt strategy, how often to alternate')
	parser.add_argument('--prim-start', type=float, default=0.0, help='What epoch to start training on the primary loss')
	parser.add_argument('--init-val', type=float, default=0.0, help='Initial Task weightings')
	parser.add_argument('--end-val', type=float, default=1.0, help='Final task weightings')
	parser.add_argument('--meta-lr-weight', type=float, default=0.01, help='learning rate for meta-learning')
	parser.add_argument('--dev_batch_sz', type=int, default=128, help='Batch sz for dev-set for meta-learning')


	parser.add_argument("--classf_warmup_frac", type=float, default=0.06)
	parser.add_argument("--classf_betas", type=str, default="(0.9,0.98)")
	parser.add_argument("--classf_dev_lr", type=float, default=1e-4, help="Learning rate of dev-head")
	parser.add_argument("--classf_wd", type=float, default=0.1)
	parser.add_argument("--classf_dev_wd", type=float, default=0.1)
	parser.add_argument("--base_wd", type=float, default=0.01)
	parser.add_argument("--classf_patience", type=int, default=5)
	parser.add_argument("--classf_max_seq_len", type=int, default=512)
	parser.add_argument("--classf_lr", type=float, default=2e-5, help="Learning rate of classifier")
	parser.add_argument("--classf_ft_lr", type=float, default=2e-6, help="Learning rate of classifier for finetuning")
	parser.add_argument("--classf_ft_iters", type=int, default=10, help='Number of finetuning iterations')
	parser.add_argument("--classf_ft_patience", type=int, default=3, help='finetuning patience iterations')
	parser.add_argument("--classf_iter_batchsz", type=int, default=8, help='Batch Size per iteration. True batch_sz is this x number of grad accumulation steps')
	parser.add_argument("--classf-metric", type=str, default='f1', choices=['f1', 'accuracy'])

	parser.add_argument("--classifier_dropout", type=float, default=0.1)
	parser.add_argument("--test_task_file", type=str, default=None)
	parser.add_argument("--dev_task_file", type=str, default=None)
	parser.add_argument("--primary_task_id", type=str, default='imdb', choices=["imdb", "amazon", "imdb_small", "citation_intent", "chemprot", "sciie"])
	parser.add_argument("--n-runs-classf", type=int, default=1)
	parser.add_argument("--only-run-classifier", action='store_true', help='Only run the classifier')
	parser.add_argument("--no-mlm-weight", action='store_true', help='Only learn the classifier - set mlm weight to 0')
	# End Change [ldery]

	parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
	parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
	parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
	args = parser.parse_args()
	args.weight_strgy = args.alpha_update_algo

	if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
		raise ValueError(
			"BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
			"flag (masked language modeling)."
		)
	if args.eval_data_file is None and args.do_eval:
		raise ValueError(
			"Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
			"or remove the --do_eval argument."
		)
	if args.should_continue:
		sorted_checkpoints = _sorted_checkpoints(args)
		if len(sorted_checkpoints) == 0:
			raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
		else:
			args.model_name_or_path = sorted_checkpoints[-1]
			print('Used Should Continue and model found is : ', args.model_name_or_path)
	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
		and not args.should_continue
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup distant debugging if needed
	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd

		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend="nccl")
		args.n_gpu = 1
	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		args.local_rank,
		device,
		args.n_gpu,
		bool(args.local_rank != -1),
		args.fp16,
	)

	# Set seed
	set_seed(args)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

	if args.config_name:
		config = AutoConfig.from_pretrained(args.config_name, output_hidden_states=True, cache_dir=args.cache_dir)
	elif args.model_name_or_path:
		config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir)
	else:
		# When we release a pip version exposing CONFIG_MAPPING,
		# we can do `config = CONFIG_MAPPING[args.model_type]()`.
		raise ValueError(
			"You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
			"and load it from here, using --config_name"
		)

	if args.tokenizer_name:
		tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
	elif args.model_name_or_path:
		tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
	else:
		raise ValueError(
			"You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
			"and load it from here, using --tokenizer_name"
		)

	if args.block_size <= 0:
		args.block_size = tokenizer.model_max_length
		# Our input block size will be the max possible for the model
	else:
		args.block_size = min(args.block_size, tokenizer.model_max_length)

	if len(args.aux_task_names) > 0:
		# we have multiple tasks. We need to have multiple lm heads
		setattr(config, 'head_names', args.aux_task_names)
		setattr(config, 'primary_head', args.aux_task_names[0])
	if args.model_name_or_path:
		model = AutoModelWithLMHead.from_pretrained(
			args.model_name_or_path,
			from_tf=bool(".ckpt" in args.model_name_or_path),
			config=config,
			cache_dir=args.cache_dir,
		)
	else:
		logger.info("Training new model from scratch")
		model = AutoModelWithLMHead.from_config(config)
	model_name = args.model_name_or_path
	assert model_name, 'The name of the model is not Set. Maybe use roberta-base as the default'
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Setting up the dataset
	train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

	# Instantiate the model with the auxiliary tasks
	logger.info("Instantiating AuxTaskModel")
	base_task_dataset_files = get_auxtask_files(args.primary_task_id)
	auxTaskModel = ModelWithAuxTasks(
										model_name, model, base_task_dataset_files, max_seq_len=args.classf_max_seq_len,
										alpha_generator_algo=args.alpha_update_algo, primary_task_id=args.primary_task_id,
										dropout=args.classifier_dropout, prim_test_file=args.test_task_file, batch_sz=args.classf_iter_batchsz,
										prim_dev_file=args.dev_task_file, save_path=os.path.join(args.output_dir, 'modelWAuxTasks.pth'),
										grad_accum_factor=args.gradient_accumulation_steps, no_mlm_weight=args.no_mlm_weight,
										dev_batch_sz=args.dev_batch_sz
									)
	# Move the model to the appropriate device
	model.to(args.device)
	auxTaskModel.to(args.device)

	if args.local_rank == 0:
		# End of barrier to make sure only the first process in distributed training download model & vocab
		torch.distributed.barrier()

	logger.info("Training/evaluation parameters %s", args)

	# Training
	if args.do_train:
		logger.info("Getting dataset")
		if args.local_rank not in [-1, 0]:
			# Barrier to make sure only the first process in distributed training process the dataset,
			# and the others will use the cache
			torch.distributed.barrier()

		if args.local_rank == 0:
			torch.distributed.barrier()
		
		logger.info("Run Training")
		if not args.only_run_classifier:
			global_step, tr_loss = train(args, train_dataset, model, tokenizer, auxTaskModel=auxTaskModel)
			logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
			# Saving the visualization
			try:
				auxTaskModel.alpha_generator_algo.viz_results(args.output_dir, group_aux=(not args.meta_learn_aux))
			except:
				logger.info("Unable to generate result visualization")
		# We want to now do the training for aux-model-independently
			# multi-gpu training (should be after apex fp16 initialization)
		if not args.no_final_finetuning:
			all_f1s, all_accs = [], []
			auxTaskModel.load_primary(args.device)
			test_metrics = auxTaskModel.evaluate_classifier(set_='test')
			dev_metrics = auxTaskModel.evaluate_classifier(set_='dev')
			print('Before Training. Dev  (F1={:.3f}, Accuracy={:.3f})'.format(dev_metrics['f1'], dev_metrics['accuracy']))
			print('Before Training. Test (F1={:.3f}, Accuracy={:.3f})'.format(test_metrics['f1'], test_metrics['accuracy']))
			no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"]
			for i in range(args.n_runs_classf):
				torch.cuda.empty_cache()
				args.seed = i
				set_seed(args)
				print('Currently working on seed : {}/{}'.format(i + 1,  args.n_runs_classf))
				print('Loading the saved model that performed best on primary task')
				auxTaskModel.load_primary(args.device)
				# Setup an optimizer for the classifier
				classifier_params = auxTaskModel.get_classifier_params(keys=[auxTaskModel.primary_task_id], withbase=True)
				optimizer_grouped_parameters = [
					{
						"params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
						"weight_decay": args.weight_decay,
					},
					{"params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
				]
				this_optim = AdamW(
											optimizer_grouped_parameters, betas=eval(args.classf_betas),
											weight_decay=args.classf_wd, lr=args.classf_ft_lr
										)

				# Todo [ldery] - maybe you should put back lr scheduling. Do hyperparam on this
				if args.n_gpu > 1 and args.parallelize_classifiers:
					auxTaskModel.parallelize_classifiers()

				this_lr_scheduler = None
				best_f1, best_acc, perfs  = auxTaskModel.train_primary(args.classf_ft_iters, this_optim, this_lr_scheduler, args.max_grad_norm, patience=args.classf_ft_patience)
				print('Run {}. Final Test (F1={:.3f}, Accuracy={:.3f})'.format(i, best_f1, best_acc))
				pickle.dump(perfs, open(os.path.join(args.output_dir, 'ftmodel.{}.perf.pkl'.format(i)), 'wb') )
				all_f1s.append(best_f1)
				all_accs.append(best_acc)

			all_accs, all_f1s = np.array(all_accs), np.array(all_f1s)
			print("Test F1 - {:3f} +/ {:.3f}".format(all_f1s.mean(), all_f1s.std()))
			print("Test Ac - {:3f} +/ {:.3f}".format(all_accs.mean(), all_accs.std()))
			pickle.dump([all_accs, all_f1s], open(os.path.join(args.output_dir, 'ftmodel_{}.bestperfs.pkl'.format(args.seed)), 'wb') )

	# Saving best-practices: if you use save_pretrained for the model and tokenizer,
	# you can reload them using from_pretrained()
	if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		# Create output directory if needed
		if args.local_rank in [-1, 0]:
			os.makedirs(args.output_dir, exist_ok=True)

		logger.info("Saving model checkpoint to %s", args.output_dir)
		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		model_to_save = (
							model.module if hasattr(model, "module") else model
		)  # Take care of distributed/parallel training
		model_to_save.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

		# Load a trained model and vocabulary that you have fine-tuned
		model = AutoModelWithLMHead.from_pretrained(args.output_dir)
		tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
		model.to(args.device)

# 	# Evaluation
	results = {}
# 	if args.do_eval and args.local_rank in [-1, 0]:
# 		checkpoints = [args.output_dir]
# 		if args.eval_all_checkpoints:
# 			checkpoints = list(
# 				os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
# 			)
# 			logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
# 		logger.info("Evaluate the following checkpoints: %s", checkpoints)
# 		for checkpoint in checkpoints:
# 			global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
# 			prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

# 			model = AutoModelWithLMHead.from_pretrained(checkpoint)
# 			model.to(args.device)
# 			result = evaluate(args, model, tokenizer, prefix=prefix)
# 			result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
# 			results.update(result)

	return results


if __name__ == "__main__":
	main()
