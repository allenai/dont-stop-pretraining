from nltk import tokenize as nltktokenize
import os
import logging
import json
import unicodedata
import numpy as np
from collections import Counter, defaultdict
from data_utils import *

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
logger = logging.getLogger(__name__)

OUT_PAD = -100

def add_data_args(parser):
	parser.add_argument('-task-data', type=str, default=None)
	parser.add_argument('-in-domain-data', type=str, default=None)
	parser.add_argument('-out-domain-data', type=str, default=None)
	parser.add_argument('-neural-lm-data', type=str, default=None)

# Todo [ldery] - will need to incorporate caching here
class LineByLineRawTextDataset(Dataset):
	def __init__(self, file_path, tokenizer, tf_or_idf_present, cap_present):
		assert os.path.isfile(file_path)
		logger.info("Creating Raw Line x Line Dataset %s", file_path)
		
		all_lines = []
		all_tokens = []
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
			freq_cntr, lines, tokens, caps = self.process_single_file(file_path, tokenizer, tf_or_idf_present, cap_present)
			all_tokens.extend(tokens)
			all_lines.extend(lines)
			self.doc_lens.append(len(lines))
			if cap_present:
				all_caps.extend(caps)
			if tf_or_idf_present:
				self.doc_tfs[doc_id] = scale(freq_cntr, smoothfactor=1)
				doc_tfs.update(list(freq_cntr.keys()))

		self.examples = all_lines
		self.tokens = all_tokens
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
			doc_tfidfs[k] = scale(new_tfidf, smoothfactor=1)
		return doc_tfidfs

	def process_single_file(self, file_path, tokenizer, tf_or_idf_present, cap_present):
		with open(file_path, encoding="utf-8") as f:
			token_counter = Counter() if tf_or_idf_present else None
			lines = []
			all_tokens = []
			capitalizations = [] if cap_present else None
			for line in f.readlines():
				line = line.strip()
				lines.append(line)
				tokens = tokenizer.tokenize(line)
				if cap_present:
					caps = get_caps(run_strip_accents(line), tokens)
					assert len(tokens) == len(caps)
					capitalizations.append(caps)
				if tf_or_idf_present:
					token_counter.update(tokens)
				all_tokens.append(tokens)
		return token_counter, lines, all_tokens, capitalizations

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		return {'sample': self.examples[i], 'idx': i}

	def getdocid(self, sent_idx):
		idx = np.searchsorted(self.doc_lens, sent_idx, side="left")
		if self.doc_lens[idx] > sent_idx:
			return self.doc_names[idx - 1]
		return self.doc_names[idx]

	def _pad_specials(self, sent_, special_tok_mask):
		if len(sent_) == len(special_tok_mask):
			assert sum(special_tok_mask) > 0, 'Sentence and token mask same length but there are special tokens'
			return torch.tensor(sent_)
		assert len(sent_) < len(special_tok_mask), 'Sentence must have fewer tokens than the special tokens mask'
		mul_ = 1.0 if isinstance(sent_[0], float) else 1
		new_sent_ = torch.full((len(special_tok_mask), ), OUT_PAD * mul_) # We only compute loss on masked tokens
		j = 0
		for idx_, tok in enumerate(special_tok_mask):
			if tok == 1:  # We have a special token here
				continue
			new_sent_[idx_] = sent_[j]
			j += 1
		assert j == len(sent_), 'New sentence not correctly filled to completion'
		return new_sent_

	def getcaps(self, sent_idx, special_token_mask):
		return self._pad_specials(self.caps[sent_idx], special_token_mask)

	def gettfs(self, sent_idx, special_token_mask):
		sent_ = self.tokens[sent_idx]
		tf_cntr = self.doc_tfs[self.getdocid(sent_idx)]
		tfs = [tf_cntr[x] for x in sent_]
		return self._pad_specials(tfs, special_token_mask)

	def gettfidfs(self, sent_idx, special_token_mask):
		sent_ = self.tokens[sent_idx]
		tfidf_cntr = self.doc_tfidfs[self.getdocid(sent_idx)]
		tfidfs = [tfidf_cntr[x] for x in sent_]
		return self._pad_specials(tfidfs, special_token_mask)

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

import pdb
class DataTransformAndItr(object):
	def __init__(self, args, dataoptions, input_tform_dict, output_dict):
		# Sets the total batch size
		self.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
		self.dataOpts = dataoptions
		self.input_tform_dict = input_tform_dict
		self.output_dict = output_dict
		self.proba = args.mlm_probability
		self.block_size = args.block_size
	
	def apply_in_tform(self, sent, in_tform_type):
		return mask_tokens(sent, self.dataOpts.tokenizer, self.proba, in_tform_type)

	def apply_out_tform(
							self, output_type, ds, padded_sent, tformed_sent,
							orig_samples, tokenID_samples, special_tok_mask
						):
		if output_type == 'DENOISE':
			assert padded_sent.shape == tformed_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tformed_sent}
		elif output_type == 'TFIDF':
			tfidf_sent = [ds.gettfidfs(x['idx'], special_tok_mask[id_]) for id_, x in enumerate(orig_samples)]
			tfidf_sent = pad_sequence(tfidf_sent, OUT_PAD)
			assert padded_sent.shape == tfidf_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tfidf_sent}
		elif output_type == 'TF':
			tf_sent = [ds.gettfs(x['idx'], special_tok_mask[id_]) for id_, x in enumerate(orig_samples)]
			tf_sent = pad_sequence(tf_sent, OUT_PAD)
			assert padded_sent.shape == tf_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tf_sent}
		elif output_type == 'CAP':
			cap_sent = [ds.getcaps(x['idx'], special_tok_mask[id_]) for id_, x in enumerate(orig_samples)]
			cap_sent = pad_sequence(cap_sent, OUT_PAD)
			assert padded_sent.shape == cap_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': cap_sent}
		elif  output_type == 'NSP':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'QT':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'FS':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'ASP':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'SO':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'SO':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		elif  output_type == 'SCP':
			# Todo[ldery] - implement
			raise NotImplementedError('This output type [{}] has not been impemented yet'.format(output_type)) 
		else:
			raise ValueError('Illegal value for output transform : {}'.format(self.output_type))

	def get_iterator(self, loss_config, shuffle=True, local_rank=-1):
		ds_id = loss_config[0] # The first stage in the loss config is the dataset-id
		# Dataset Obtained
		ds = self.dataOpts.get_dataset(ds_id)
		in_tform = self.input_tform_dict[loss_config[1]]
		out_tform = self.output_dict[loss_config[-1]]
		def collate(examples):
			all_egs = [x['sample'] for x in examples]
			# Need to be careful because encode_plus adds special tokens
			# Todo [ldery] Need to reconsider whether doing this here makes things super slow
			out = self.dataOpts.tokenizer.batch_encode_plus(
									all_egs, add_special_tokens=True, truncation=True,
									max_length=self.block_size, return_special_tokens_mask=True
					)
			all_egs = [torch.tensor(x) for x in out["input_ids"]]
			special_tok_mask = out["special_tokens_mask"]
			inputs = pad_sequence(all_egs, self.dataOpts.tokenizer.pad_token_id)
			# need to do the outputs
			inputs, labels, _ = self.apply_in_tform(inputs, in_tform)
			return self.apply_out_tform(out_tform, ds, inputs, labels, examples, all_egs, special_tok_mask)

		sampler = RandomSampler(ds) if local_rank == -1 else DistributedSampler(ds)
		dataloader = DataLoader(
			ds, sampler=sampler, batch_size=self.train_batch_size, collate_fn=collate
		)
		return dataloader


# Todo  [ldery] - run some tests to make sure code here is working
def run_tests():
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
		print("test datasets match config: {}".format(msg))
		exit()
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
		exit()
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
		exit()
	print("test correct doc ids: {}".format(msg))
	print('==='*5, 'DataOptions Tests Passed. Moving on to DataTransformAndItr', '==='*5)
	setattr(args, 'per_gpu_train_batch_size', 32)
	setattr(args, 'n_gpu', 1)
	setattr(args, 'mlm_probability', 0.15)
	setattr(args, 'block_size', 512)

	try:
		dtform_and_itr = DataTransformAndItr(args, aux_dataOptions, autoloss_config.get_stage(1), autoloss_config.get_stage(-1))
		msg = 'Passed.'
	except:
		msg = 'Failed.'
		print("test Init DataTransformAndItr: {}".format(msg))
		exit()
	print("test Init DataTransformAndItr: {}".format(msg))
	# Main approach to testing this was checking if the outputs look reasonable
	print('Input Should be same as output for chosen indices')
	loss_config = (1, 0, 0, 0)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	
	print('Input Should be different token as output for chosen indices')
	loss_config = (1, 1, 0, 0)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	
	print('Input Should be have mask token as output for chosen indices')
	loss_config = (1, 2, 0, 0)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	print('Input Should be have mask token as output for chosen indices and tfids for output')
	loss_config = (1, 2, 0, 1)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	pdb.set_trace()

	print('Input Should be have mask token as output for chosen indices and capitalization for output')
	loss_config = (1, 2, 0, 3)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')
	pdb.set_trace()

	print('Input Should be have mask token as output for chosen indices and term-frequencies for output')
	loss_config = (1, 2, 0, 2)
	itr_ = dtform_and_itr.get_iterator(loss_config)
	for k in itr_:
		inp, out = k['input'], k['output']
	print('Visually inspected and works')


if __name__ == '__main__':
	run_tests()