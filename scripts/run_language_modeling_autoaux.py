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
	AutoModelWithLMHead,
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
	get_linear_schedule_with_warmup,
)

from ..AutoSearchSpace.config import add_config_args, Config
from ..AutoSearchSpace.data import add_data_args, DataOptions

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
				self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
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


class LineByLineTextDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
		assert os.path.isfile(file_path)
		# Here, we do not cache the features, operating under the assumption
		# that we will soon use fast multithreaded tokenizers from the
		# `tokenizers` repo everywhere =)
		logger.info("Creating features from dataset file at %s", file_path)

		with open(file_path, encoding="utf-8") as f:
			lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

		self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
	file_path = args.eval_data_file if evaluate else args.train_data_file
	if args.line_by_line:
		return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
	else:
		return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


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
	return tokenizer

def auto_auxiliary(args):
	# Create the configuration object
	autoloss_config = Config(args.searchspace_config)

	# Create the datasets based on the configuration
	tokenizer = get_tokenizer(args)
	# Taking that the config stage 0 is the input stage
	aux_dataOptions = DataOptions(args, tokenizer, autoloss_config.get_stage(0), autoloss_config.get_stage(-1))
	# Create the data transform and iterator
	model = DataTransformAndItr(args, aux_dataOptions, autoloss_config.get_stage(1), autoloss_config.get_stage(-1))

	# Generate the loss functions based on the configuration

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

	if args.config_name:
		config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
	elif args.model_name_or_path:
		config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
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
			config=config,
			cache_dir=args.cache_dir,
		)
	else:
		logger.info("Training new model from scratch")
		model = AutoModelWithLMHead.from_config(config)

	model.to(args.device)

	
	
	if args.local_rank not in [-1, 0]:
			torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
	train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
	
	if args.local_rank == 0:
		torch.distributed.barrier()
	
	global_step, tr_loss = train(args, train_dataset, model, tokenizer)
	logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
	
	# Todo [ldery] - remove when the model is returned
	return model


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
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
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
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
	parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
	parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
	parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
	
	# ldery - for adding more arguments as appropriate
	add_config_args(parser)
	args = parser.parse_args()

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

	if args.local_rank == 0:
		torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

	logger.info("Training/evaluation parameters %s", args)

	# Training
	if args.do_train:
		model = auto_auxiliary(args)

	# Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
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

		# Todo [ldery] - modify this to fix it
		# Load a trained model and vocabulary that you have fine-tuned
		model = AutoModelWithLMHead.from_pretrained(args.output_dir)
		tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
		model.to(args.device)

if __name__ == "__main__":
	main()
