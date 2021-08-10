from nltk import tokenize as nltktokenize
from torch.nn.utils.rnn import pad_sequence
import os
import logging
import json
import unicodedata
import numpy as np
from collections import Counter, defaultdict
from data_utils import *

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
logger = logging.getLogger(__name__)

def add_data_args(parser):
	parser.add_argument('-task-data', type=str, default=None)
	parser.add_argument('-in-domain-data', type=str, default=None)
	parser.add_argument('-out-domain-data', type=str, default=None)
	parser.add_argument('-neural-lm-data', type=str, default=None)


class LineByLineRawTextDataset(Dataset):
	def __init__(self, file_path, tokenizer, tf_or_idf_present, cap_present):
		assert os.path.isfile(file_path)
		logger.info("Creating Raw Line x Line Dataset %s", file_path)
		
		all_lines = []
		self.doc_lens = []
		self.doc_names = []

		if cap_present:
			all_caps = []
		if tf_or_idf_present:
			doc_tfs = Counter()
			self.doc_tfs = {}
	
		if '.json' in file_path:
			docs = json.load(open(file_path, 'r'))
		else:
			docs = {'0': file_path}
		# Process each document
		self.doc_lens.append(0)
		for doc_id, file_path in docs.items():
			self.doc_names.append(doc_id)
			# Get the lines, token frequencies and capitalization
			freq_cntr, lines, caps = self.process_single_file(file_path, tokenizer, tf_or_idf_present, cap_present)
			all_lines.extend(lines)
			self.doc_lens.append(len(lines))
			if cap_present:
				all_caps.extend(caps)
			if tf_or_idf_present:
				self.doc_tfs[doc_id] = scale(freq_cntr)
				doc_tfs.update(list(freq_cntr.keys()))

		self.examples = all_lines
		self.doc_lens = np.cumsum(self.doc_lens)

		if tf_or_idf_present:
			# need to do some idf computation here
			smoothed_n = 1 + len(docs)
			doc_idfs = {x : np.log(smoothed_n/(1 + v) + 1.0) for x, v in doc_tfs.items()}
			self.doc_tfidfs = self.get_tfidfs(self.doc_tfs, doc_idfs)
		if cap_present:
			self.caps = all_caps
			assert len(self.caps) == len(self.examples), 'The number of caps should match the number of example sentences'


	def get_tfidfs(self, doc_tfs, doc_idfs):
		doc_tfidfs = defaultdict(lambda:defaultdict(float))
		for k, tf in doc_tfs.items():
			new_tfidf = {x: v * doc_idfs[x] for x, v in tf.items()}
			doc_tfidfs[k] = scale(new_tfidf)
		return doc_tfidfs

	def process_single_file(self, file_path, tokenizer, tf_or_idf_present, cap_present):
		with open(file_path, encoding="utf-8") as f:
			token_counter = Counter() if tf_or_idf_present else None
			lines = []
			capitalizations = [] if cap_present else None
			for line in f.readlines():
				tokens = tokenizer.tokenize(line)
				if cap_present:
					caps = get_caps(run_strip_accents(line), tokens)
					assert len(tokens) == len(caps)
					capitalizations.append(caps)
				if tf_or_idf_present:
					token_counter.update(tokens)
				lines.append(tokens)
		return token_counter, lines, capitalizations

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		return {'sample': self.examples[i], 'idx': i}

	def getdocid(self, sent_idx):
		idx = np.searchsorted(self.doc_lens, sent_idx, side="left")
		if self.doc_lens[idx] > sent_idx:
			return self.doc_names[idx - 1]
		return self.doc_names[idx]

# Todo[ldery] - will need to incorporate options for target computation here
class DataOptions(object):
	def __init__(self, args, tokenizer, data_dict, output_dict):
		self.construct_dataset_map(args, data_dict, output_dict, tokenizer)
		self.tokenizer = tokenizer

	def construct_dataset_map(self, args, data_dict, output_dict, tokenizer):
		self.id_to_dataset_dict = {}
		tf_or_idf_present = ('TFIDF' in output_dict.values()) or ('TF' in output_dict.values())
		cap_present = 'CAP' in output_dict.values()
		for v, k in data_dict.items():
			path = None
			if k == 'Task':
				assert args.task_data is not None, 'Task Data Location not specified'
				path = args.task_data
			elif k == 'In-Domain':
				assert args.in_domain_data is not None, 'In Domain Data Location not specified'
				path = args.in_domain_data
			elif k == 'Out-Domain':
				assert args.out_domain_data is not None, 'In Domain Data Location not specified'
				path = args.out_domain_data
			elif k == 'Neural-LM':
				assert args.neural_lm_data is not None, 'In Domain Data Location not specified'
				path = args.neural_lm_data
			assert path is not None, 'Invalid data type given. {}:{}'.format(k, v)
			dataset = LineByLineRawTextDataset(path, tokenizer, tf_or_idf_present, cap_present)
			self.id_to_dataset_dict[v] = dataset

	def get_dataset(self, id_):
		return self.id_to_dataset_dict[id_]


class DataTransformAndItr(object):
	def __int__(self, args, dataoptions, input_tform_dict, output_dict):
		self.set_max_sent_len(args)
		# Sets the total batch size
		self.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
		self.dataOpts = dataoptions
		self.input_tform_dict = input_tform_dict
		self.output_dict = output_dict
		self.proba = args.mlm_probability
	
	def apply_in_tform(self, sent, in_tform_type):
		return mask_tokens(sent, self.dataOpts.tokenizer, self.proba, in_tform_type)

	def apply_out_tform(self, orig_sent, tformed_sent, orig_samples, enc_samples):
		if self.output_type == 'DENOISE':
			return {'input': orig_sent, 'output:', tformed_sent}
		elif self.output_type == 'TFIDF':
			tfidf_sent = 
			return {'input': orig_sent, 'output:', tfidf_sent}
		elif self.output_type == 'TF':
			tf_sent = 
			return {'input': orig_sent, 'output:', tf_sent}
		elif self.output_type == 'CAP':
			cap_sent
			return {'input': orig_sent, 'output:', cap_sent}
		elif  self.output_type == 'NSP':
		elif  self.output_type == 'QT':
			# Todo[ldery] - implement
			pass
		elif  self.output_type == 'FS':
			# Todo[ldery] - implement
			pass
		elif  self.output_type == 'ASP':
			# Todo[ldery] - implement
			pass
		elif  self.output_type == 'SO':
			# Todo[ldery] - implement
			pass
		elif  self.output_type == 'SO':
			# Todo[ldery] - implement
			pass
		elif  self.output_type == 'SCP':
			# Todo[ldery] - implement
			pass
		else:
			raise ValueError('Illegal value for output transform : {}'.format(self.output_type))

	def get_iterator(self, loss_config, shuffle=True):
		ds_id = loss_config[0]
		# Dataset Obtained
		ds = self.dataOpts.get_dataset(ds_id)
		in_tform = self.input_tform_dict[loss_config[1]]
		out_tform = self.output_dict[loss_config[-1]]

		def collate(examples):
			all_egs = [x['sample'] for x in examples]
			# Need to be careful because encode_plus adds special tokens
			all_egs = self.dataOpts.tokenizer.batch_encode_plus(all_egs, add_special_tokens=True, max_length=self.block_size)["input_ids"]
			if self.dataOpts.tokenizer._pad_token is None:
				inputs = pad_sequence(all_egs, batch_first=True)
			else:
				inputs = pad_sequence(all_egs, batch_first=True, padding_value=self.dataOpts.tokenizer.pad_token_id)
			# need to do the outputs
			inputs, labels, _ = self.apply_in_tform(inputs, in_tform)
			return self.apply_out_tform(inputs, labels, examples, all_egs)

		sampler = RandomSampler(ds) if args.local_rank == -1 else DistributedSampler(ds)
		dataloader = DataLoader(
			ds, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
		)
		return dataloader

	def set_max_sent_len(self, args):
		if args.block_size <= 0:
			self.block_size = dataoptions.tokenizer.max_len
		# Our input block size will be the max possible for the model
		else:
			self.block_size = min(args.block_size, dataoptions.tokenizer.max_len)



# Todo  [ldery] - run some tests to make sure code here is working
def run_dataoptions_tests():
	import pdb
	import argparse
	from config import Config
	from transformers import (
		AutoTokenizer,
		PreTrainedTokenizer,
	)
	parser = argparse.ArgumentParser()
	add_data_args(parser)
	args = parser.parse_args()
	args.task_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/citation_intent.dev.txt'
	args.in_domain_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/test.json'
	print('Getting Config')
	autoloss_config = Config('full')
	print('Getting Tokenizer')
	tokenizer = AutoTokenizer.from_pretrained('roberta-base')
	# Taking that the config stage 0 is the input stage
	print('Getting aux_dataOptions')
	try:
		print('Some config entries not present. Should throw error')
		aux_dataOptions = DataOptions(args, tokenizer, autoloss_config.get_stage(0), autoloss_config.get_stage(-1))
		msg = 'Failed.'
	except:
		msg = 'Passed.'
	print("test datasets match config: {}".format(msg))
	args.out_domain_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/sciie.dev.txt'
	args.neural_lm_data = '/home/ldery/projects/AutoAuxiliaryLoss/test_datasets/chemprot.dev.txt'
	
	try:
		print('Able to load full options and process all data')
		aux_dataOptions = DataOptions(args, tokenizer, autoloss_config.get_stage(0), autoloss_config.get_stage(-1))
		msg = 'Passed.'
	except:
		msg = 'Failed.'
	print("test can load data option: {}".format(msg))

	try:
		print('Checking doc ids')
		ds = aux_dataOptions.get_dataset(1)
		assert ds.getdocid(0) == 'CHEMPROT', 'Checking correct doc_id for 0'
		assert ds.getdocid(2442) == 'CITATION', 'Checking correct doc_id for 2542'
		assert ds.getdocid(2542) == 'SCIIE', 'Checking correct doc_id for 2542'
		assert ds.getdocid(3109) == 'CITATION.1', 'Checking correct doc_id for 3109'
		msg = 'Passed.'
	except:
		msg = 'Failed.'
	print("test correct doc ids: {}".format(msg))

if __name__ == '__main__':
	run_dataoptions_tests()