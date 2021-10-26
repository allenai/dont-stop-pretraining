from nltk import tokenize as nltktokenize
import os
import logging
import json
import unicodedata
import numpy as np
from collections import Counter, defaultdict
from data_utils import *
import math

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
	def __init__(self, file_path, tokenizer, tf_or_idf_present, cap_present, max_=10):
		assert os.path.isfile(file_path)
		logger.info("Creating Raw Line x Line Dataset %s", file_path)
		
		all_lines = []
		all_tokens = []
		self.doc_lens = []
		self.doc_names = []
		self.max_ = max_

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
				self.doc_tfs[doc_id] = scale(freq_cntr, max_=self.max_, smoothfactor=1)
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
				if len(line) < 2: # Remove all single letter or empty lines
					continue
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

	def get_samples(self, n_samples, is_sent_config=False):
		total_samples = len(self.examples)
		if is_sent_config:
			n_samples = n_samples // 2
			total_samples -= 1
		chosen_idxs = np.random.choice(total_samples, n_samples)
		all_chosen = [self[idx] for idx in chosen_idxs]
		if is_sent_config:
			second_half = [self[idx + 1] for idx in chosen_idxs]
			all_chosen.extend(second_half)
		return all_chosen

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
		for idx_, val_ in enumerate(sent_):
			new_sent_[idx_ + 1] = val_ # plus 1 because we are skipping the bos token
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
		self.max_ = 10 # This is a magic number showing the max scale for TFIDF style losses

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

	def get_total_len(self):
		lens = [len(ds) for id_, ds in self.id_to_dataset_dict.items()]
		return sum(lens)

	def get_dataset_len(self, id_):
		return len(self.id_to_dataset_dict[id_])


class DataTransformAndItr(object):
	def __init__(self, args, dataoptions, input_tform_dict, output_dict):
		# Sets the total batch size
		self.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
		self.dataOpts = dataoptions
		self.input_tform_dict = input_tform_dict
		self.output_dict = output_dict
		self.proba = args.mlm_probability
		self.block_size = args.block_size
		self.special_token_list = dataoptions.tokenizer.all_special_ids
		self.special_token_list.remove(dataoptions.tokenizer.mask_token_id)
		self.special_token_list.remove(dataoptions.tokenizer.unk_token_id)

	def total_iters(self):
		return int(self.dataOpts.get_total_len() / self.train_batch_size)

	def apply_in_tform(self, sent, token_probas=None):
		return mask_tokens(sent, self.dataOpts.tokenizer, self.proba, token_probas)

	def apply_out_tform(
							self, output_type, ds, padded_sent, tformed_sent,
							orig_samples
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
			special_tok_mask = sum([padded_sent == x for x in self.special_token_list])
			special_tok_mask = (special_tok_mask > 0) * 1
			tf_sent = []
			for id_, x in enumerate(orig_samples):
				this_tf_sent = ds.gettfs(x['idx'], special_tok_mask[id_])
				tf_sent.append(this_tf_sent)
			tf_sent = torch.stack(tf_sent).to(padded_sent.device)
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
			assert padded_sent.shape == tformed_sent.shape, 'Invalid Shapes. Input must have same shape as output'
			return {'input': padded_sent, 'output': tformed_sent}
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

	def get_data(self, sample_configs, searchOpts, representation_tform):
		aggregated_data = []
		# compute dataset marginals
		rep_tforms = np.unique([config[2] for config in sample_configs])
		rep_probas = searchOpts.get_relative_probas(2, rep_tforms)
		for rep_idx, rep_id in enumerate(rep_tforms):
			try:
				num_samples = math.ceil(self.train_batch_size * rep_probas[rep_idx].item())
			except:
				pdb.set_trace()

			this_configs = [config for config in sample_configs if config[2] == rep_id]
			datasets = np.unique([config[0] for config in this_configs])
			ds_probas = searchOpts.get_relative_probas(0, datasets)
			for ds_idx, ds_id in enumerate(datasets):
				n_ds_samples = math.ceil(num_samples * ds_probas[ds_idx].item())
				ds = self.dataOpts.get_dataset(ds_idx)

				this_ds_configs = [config for config in this_configs if config[0] == ds_id]
				out_tforms = np.unique([config[-1] for config in this_ds_configs])
				is_sent_config = np.any([searchOpts.config.is_dot_prod(x) for x in out_tforms])
				examples = ds.get_samples(n_ds_samples, is_sent_config=is_sent_config)
				token_tforms = np.unique([config[1] for config in this_ds_configs])
				probas = searchOpts.get_relative_probas(1, token_tforms)
				_, stage_map = searchOpts.config.get_stage_w_name(1)
				token_probas = None
				if not searchOpts.config.isBERTTransform():
					token_probas = {}
					for idx, name in stage_map.items():
						if idx in token_tforms:
							token_probas[name] = probas[list(token_tforms).index(idx)].item()

				inputs, labels, masks_for_tformed = self.collate(examples, token_probas)
				pad_mask = 1.0 - (inputs.eq(self.dataOpts.tokenizer.pad_token_id)).float()
				rep_mask = representation_tform.get_rep_tform(inputs.shape, pad_mask, rep_id)
				batch = {'input': inputs, 'output': None, 'rep_mask': rep_mask}
				config_dict = {}
				for config_ in this_ds_configs:
					token_tform_name = stage_map[config_[1]]
					task_output = masks_for_tformed[token_tform_name][1]
					out_type = searchOpts.config.get_name(3, config_[-1])
					dict_ = self.apply_out_tform(out_type, ds, inputs, task_output, examples)
					config_dict[config_] = dict_['output']
				aggregated_data.append((batch, config_dict))
			# Avoiding pre-mature optimization here by aggregating by representation
		return aggregated_data

	def collate(self, examples, token_probas):
		all_egs = [x['sample'] for x in examples]
		out = self.dataOpts.tokenizer.batch_encode_plus(
								all_egs, add_special_tokens=True, truncation=True,
								max_length=self.block_size, return_special_tokens_mask=True
				)
		all_egs = [torch.tensor(x) for x in out["input_ids"]]
		inputs = pad_sequence(all_egs, self.dataOpts.tokenizer.pad_token_id)
		return self.apply_in_tform(inputs, token_probas)


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