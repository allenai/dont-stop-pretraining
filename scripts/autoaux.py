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
from typing import Dict, List, Tuple
import pdb
import gc

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
	MODEL_WITH_LM_HEAD_MAPPING,
	WEIGHTS_NAME,
	AdamW,
	AutoConfig,
	AutoModel,
	AutoModelWithLMHead,
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
	get_linear_schedule_with_warmup,
)

import sys
PATH = os.path.join(os.getcwd(), "AutoSearchSpace/")

sys.path.insert(1, PATH)

from config import add_config_args, Config
from data import add_data_args, DataOptions, DataTransformAndItr
from searchspace import SearchOptions, add_searchspace_args
from representation_mask import RepTransform
from modelling import ModelWithAuxTasks, add_modelling_options


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter


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
	if scheduler is not None:
		torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
	logger.info("Saving optimizer and scheduler states to %s", output_dir)


def get_tokenizer(args):
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
	return tokenizer

def auto_auxiliary(args):
	# Create the configuration object
	autoloss_config = Config(args.searchspace_config)

	# Create the datasets based on the configuration
	tokenizer = get_tokenizer(args)

	# Taking that the config stage 0 is the input stage
	aux_dataOptions = DataOptions(args, tokenizer, autoloss_config.get_stage(0), autoloss_config.get_stage(-1))

	# Create the data transform and iterator
	dtform_and_itr = DataTransformAndItr(args, aux_dataOptions, autoloss_config.get_stage(1), autoloss_config.get_stage(-1))

	# Create the search options object
	searchOpts = SearchOptions(
									autoloss_config, args.prim_aux_lr, args.auxiliaries_lr, use_EG=args.use_EG, step_every=args.step_meta_every,
									use_factored_model=args.use_factored_model, is_cuda=True, token_temp=args.token_temp
								)

	# enumerate the valid loss configs and get iterators for each loss type
	max_dataset_len = dtform_and_itr.total_iters()
	representation_tform = RepTransform(autoloss_config.get_stage(2))

	# Instantiate the model
	if args.config_name:
		model_config = AutoConfig.from_pretrained(args.config_name, output_hidden_states=True, cache_dir=args.cache_dir)
	elif args.model_name_or_path:
		model_config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir)
	else:
		# When we release a pip version exposing CONFIG_MAPPING,
		# we can do `config = CONFIG_MAPPING[args.model_type]()`.
		raise ValueError(
			"You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
			"and load it from here, using --config_name"
		)
	if args.model_name_or_path:
		model = AutoModelWithLMHead.from_pretrained(
			args.model_name_or_path,
			from_tf=bool(".ckpt" in args.model_name_or_path),
			config=model_config,
			cache_dir=args.cache_dir,
		)
	else:
		logger.info("Training new model from scratch")
		model = AutoModelWithLMHead.from_config(model_config)

	# Instantiate the wrapper model before moving to device
	primary_task_info = {
			'prim_task_id': args.prim_task_id,
			'train_fname': args.train_data_file,
			'dev_fname': args.dev_data_file,
			'test_fname': args.test_data_file,
		}
	model.resize_token_embeddings(len(tokenizer))
	
	wrapper_model = ModelWithAuxTasks(
										args.model_name_or_path, model, searchOpts, args, primary_task_info,
										max_seq_len=args.classf_max_seq_len, dropout=args.classifier_dropout,
										save_path=os.path.join(args.output_dir, 'modelWAuxTasks.pth'),
										grad_accum_factor=args.gradient_accumulation_steps, batch_sz=args.classf_iter_batchsz,
										dev_batch_sz=args.dev_batch_sz, share_output_heads=args.share_output_heads
					)
	model.to(args.device)
	wrapper_model.to(args.device)

	# Setup for training the model
	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (max_dataset_len // args.gradient_accumulation_steps) + 1
		logger.info('The number of epochs is : {}'.format(args.num_train_epochs))
	else:
		t_total = max_dataset_len // args.gradient_accumulation_steps * args.num_train_epochs

	no_decay = ["bias", "LayerNorm.weight"]
	args.weight_decay = args.base_wd
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	
	# Setup the optimizer for the base model
	optimizer = AdamW(
						optimizer_grouped_parameters, betas=eval(args.classf_betas),
						lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.base_wd
					)
	scheduler = None if args.no_scheduler else get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=int(args.classf_warmup_frac * t_total), num_training_steps=t_total
	)
	
	# Setup an optimizer for the model heads
	classifier_params = wrapper_model.get_classifier_params(withbase=False)
	classifier_optim = AdamW(
								classifier_params, betas=eval(args.classf_betas),
								weight_decay=args.classf_wd, lr=args.classf_lr
							)
	classifier_scheduler = None if args.no_scheduler else get_linear_schedule_with_warmup(
		classifier_optim, num_warmup_steps=int(args.classf_warmup_frac * t_total), num_training_steps=t_total
	)
	
	# Check if saved optimizer or scheduler states exist
	if (
		args.model_name_or_path
		and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
		and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
	):
		# Load in optimizer and scheduler states
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
	
	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.per_gpu_train_batch_size
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
	
	set_seed(args)  # Added here for reproducibility
	train_iterator = trange(
		epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
	)
	early_stop = False
	classifier_dev_perfs = []
	for epoch in train_iterator:
		if early_stop:
			break

		epoch_iterator = tqdm(range(max_dataset_len), desc="Iteration")
		for iter_ in epoch_iterator:
			# Skip past any already trained steps if resuming training
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue

			# Sample a subset of the valid configurations
			sample_configs = searchOpts.sample_configurations(args.num_config_samples)
			aggregate_data = dtform_and_itr.get_data(sample_configs, searchOpts, representation_tform)

			# This does an automatic filling of the gradients. Accumulates the gradients
			try:
				wrapper_model.get_grads_with_auxiliaries(aggregate_data, searchOpts)
			except Exception as e:
				print(' | Experienced a runtime error')
				torch.cuda.empty_cache()
				gc.collect()
				if 'out of memory' in str(e):
					print('| WARNING: ran out of memory, Skipping this step')
				else:
					print('| Warning: new error experienced - this is REALLY BAD : ', e)

			if (iter_ + 1) % args.gradient_accumulation_steps == 0:
				# We have accumulated enough gradient and can now do a step
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				if args.individ_head_norm:
					for head_name in wrapper_model.head_list:
						torch.nn.utils.clip_grad_norm_(wrapper_model.get_classifier_params(keys=[head_name]), args.max_grad_norm)
				else:
					torch.nn.utils.clip_grad_norm_(wrapper_model.get_classifier_params(), args.max_grad_norm)
				
				# Step on the head parameters
				classifier_optim.step()
				if classifier_scheduler is not None:
					classifier_scheduler.step()
				classifier_optim.zero_grad()
				
				# Step on the body parameters
				optimizer.step()
				if scheduler is not None:
					scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				# reset the wrapper model dev head
				wrapper_model.reset_dev_head()
				# Update weights and also clears the gradients
				searchOpts.update_weighttensor()
				torch.cuda.empty_cache()
				
				# ask the wrapper to track the stats + tfsummarywriter so we can see in real time
				wrapper_model.push_to_tensorboard(global_step)

				# This saves only the model base
				if args.save_steps > 0 and global_step % args.save_steps == 0:
					save_chkpt(args, str(global_step), model, tokenizer, optimizer, scheduler, rotate_chkpt=True)

				if global_step % args.eval_every == 0:
					train_metrics = wrapper_model.get_metrics(reset=True)
					dev_metrics = wrapper_model.evaluate_classifier(set_='dev')
					test_metrics = wrapper_model.evaluate_classifier(set_='test')
					f1_metric_ = {
						'train': train_metrics['f1'],
						'dev': dev_metrics['f1'],
						'test': test_metrics['f1']
					}
					wrapper_model.push_metric_to_tensorboard(f1_metric_, global_step, 'primtask.f1')
					acc_metric_ = {
						'train': train_metrics['accuracy'],
						'dev': dev_metrics['accuracy'],
						'test': test_metrics['accuracy']
					}
					wrapper_model.push_metric_to_tensorboard(acc_metric_, global_step, 'primtask.acc')

					for k, v in train_metrics.items():
						print_out = "[{}] | Train : {:.3f} | Dev Set : {:.3f} | Test Set : {:.3f}".format(k, v, dev_metrics[k], test_metrics[k])
						logger.info(print_out)
					classifier_dev_perfs.append(dev_metrics[args.classf_metric])



					if dev_metrics[args.classf_metric] >= max(classifier_dev_perfs):
						# We want to save the best model here
						print('Current best dev f1 = {} achieved. Saving model'.format(dev_metrics[args.classf_metric]))
						logger.info('Now Saving the Classifier Model')
						wrapper_model.save()
						logger.info('Saving Base Model')
						save_chkpt(args, 'best', model, tokenizer, optimizer, scheduler, rotate_chkpt=False)

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
	# Close the tensorboard writer
	wrapper_model.close_writer()
	return model, wrapper_model, tokenizer, global_step



def final_finetuning(auxTaskModel, args):
	all_f1s, all_accs = [], []
	auxTaskModel.load_primary(args.device)
	test_metrics = auxTaskModel.evaluate_classifier(set_='test')
	dev_metrics = auxTaskModel.evaluate_classifier(set_='dev')
	print('Before Training. Dev  (F1={:.3f}, Accuracy={:.3f})'.format(dev_metrics['f1'], dev_metrics['accuracy']))
	print('Before Training. Test ( F1 = {:.3f} , Accuracy = {:.3f} )'.format(test_metrics['f1'], test_metrics['accuracy']))
	no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"]
	pre_ft = test_metrics
	post_ft = {'f1': None, 'accuracy': None}
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


		this_lr_scheduler = None
		best_f1, best_acc, perfs, dev_perfs  = auxTaskModel.train_primary(args.classf_ft_iters, this_optim, this_lr_scheduler, args.max_grad_norm, patience=args.classf_ft_patience, metric=args.classf_metric)
		if args.classf_metric == 'f1':
			best_f1 = best_f1 if dev_perfs[0] > dev_metrics['f1'] else test_metrics['f1']
			best_acc = best_acc if dev_perfs[0] > dev_metrics['f1'] else test_metrics['accuracy']
		else:
			best_f1 = best_f1 if dev_perfs[1] > dev_metrics['accuracy'] else test_metrics['f1']
			best_acc = best_acc if dev_perfs[1] > dev_metrics['accuracy'] else test_metrics['accuracy']

		print('Run {}. Final Test (F1={:.3f}, Accuracy={:.3f})'.format(i, best_f1, best_acc))
		pickle.dump(perfs, open(os.path.join(args.output_dir, 'ftmodel.{}.perf.pkl'.format(i)), 'wb') )
		all_f1s.append(best_f1)
		all_accs.append(best_acc)
		post_ft['f1'] = best_f1
		post_ft['accuracy'] = best_acc

	all_accs, all_f1s = np.array(all_accs), np.array(all_f1s)
	print("Test F1 - {:3f} +/ {:.3f}".format(all_f1s.mean(), all_f1s.std()))
	print("Test Ac - {:3f} +/ {:.3f}".format(all_accs.mean(), all_accs.std()))
	pickle.dump([pre_ft, post_ft], open(os.path.join(args.output_dir, 'ftmodel.bestperfs.pkl'), 'wb'))


def main():
	parser = argparse.ArgumentParser()
	
	# Required parameters
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
		"--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
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

	parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")

	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--individ_head_norm", action='store_true', help="Normalize the heads individually")
	parser.add_argument("--base_wd", type=float, default=0.01)
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

	parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
	parser.add_argument("--no-scheduler", action='store_true')
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
	parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
	parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
	parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
	
	# ldery - for adding more arguments as appropriate
	add_config_args(parser)
	add_searchspace_args(parser)
	add_data_args(parser)
	add_modelling_options(parser)
	args = parser.parse_args()

	if args.model_type in ["bert", "roberta", "distilbert", "camembert"]:
		raise ValueError(
			"BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
			"flag (masked language modeling)."
		)
	if args.should_continue:
		sorted_checkpoints = _sorted_checkpoints(args)
		if len(sorted_checkpoints) == 0:
			raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
		else:
			args.model_name_or_path = sorted_checkpoints[-1]

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

	os.makedirs(args.output_dir, exist_ok=True)

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

	if args.local_rank == 0:
		torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

	logger.info("Training/evaluation parameters %s", args)
	
	# Training
	if args.do_train:
		base_model, model, tokenizer, global_step = auto_auxiliary(args)
	if not args.no_final_finetuning:
		assert model is not None, 'The model has not been instantiated yet'
		final_finetuning(model, args)
	

	# Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
	if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		# Create output directory if needed
		if args.local_rank in [-1, 0]:
			os.makedirs(args.output_dir, exist_ok=True)

		logger.info("Saving model checkpoint to %s", args.output_dir)
		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		model_to_save = (
			model.module if hasattr(base_model, "module") else base_model
		)  # Take care of distributed/parallel training
		base_model.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
	main()
