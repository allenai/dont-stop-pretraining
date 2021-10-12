import torch
import unicodedata
import numpy as np
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
import pdb

def pad_sequence(all_egs, pad_token_id):
	if pad_token_id is None:
		return torch_pad_sequence(all_egs, batch_first=True)
	else:
		return torch_pad_sequence(all_egs, batch_first=True, padding_value=pad_token_id)


def get_probability_matrix(proba, labels, tokenizer):
	probability_matrix = torch.full(labels.shape, proba)
	special_tokens_mask = [
		tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
	]
	probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
	if tokenizer._pad_token is not None:
		padding_mask = labels.eq(tokenizer.pad_token_id)
		probability_matrix.masked_fill_(padding_mask, value=0.0)
	return probability_matrix


def mask_tokens(inputs, tokenizer, proba, token_proba=None):
	""" Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

	if tokenizer.mask_token is None:
		raise ValueError(
			"This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
		)
	labels = inputs.clone()
	# We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
	probability_matrix = get_probability_matrix(proba, labels, tokenizer)
	masked_indices = torch.bernoulli(probability_matrix).bool()
	labels[~masked_indices] = -100	# We only compute loss on masked tokens

	tformed_mask_indices = {}
	# x% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
	mask_proba = 0.8 if token_proba is None else token_proba['Mask']
	indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_proba)).bool() & masked_indices
	inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
	
	mask_output = labels.clone().detach()
	mask_output[~indices_replaced] = -100
	tformed_mask_indices['Mask'] = (indices_replaced, mask_output)

	# y% of the time, we replace masked input tokens with random word
	replace_proba = 0.5 if token_proba is None else (token_proba['Replace'] / (token_proba['Replace'] + token_proba['None']))
	indices_random = torch.bernoulli(torch.full(labels.shape, replace_proba)).bool() & masked_indices & ~indices_replaced
	random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
	inputs[indices_random] = random_words[indices_random]

	replace_output = labels.clone().detach()
	replace_output[~indices_random] = -100
	tformed_mask_indices['Replace'] = (indices_random, replace_output)

	indices_none = masked_indices & ~(indices_replaced | indices_random)
	none_output = labels.clone().detach()
	none_output[~indices_none] = -100
	tformed_mask_indices['None'] = (indices_none, none_output)
	return inputs, labels, tformed_mask_indices



# code burrowed from : 
# https://github.com/msps9341012/olfmlm/blob/master/data_utils/datasets.py
def run_strip_accents(text):
	"""Strips accents from a piece of text."""
	text = unicodedata.normalize("NFD", text)
	output = []
	for char in text:
		cat = unicodedata.category(char)
		if cat == "Mn":
			continue
		output.append(char)
	return "".join(output)

# code burrowed from : 
# https://github.com/msps9341012/olfmlm/blob/master/data_utils/datasets.py
def get_caps(s, t):
	si = 0
	ti = 0
	tii = 0
	caps = [0] * len(t)
	while si < len(s) and ti < len(t):
		if t[ti][tii] == s[si]:
			if s[si].isupper():
				caps[ti] = 1
			si += 1
			tii += 1
		elif t[ti][tii] == "#" or ord(t[ti][tii]) >= 128:
			tii += 1
		elif s[si] in [" ", "#"] or ord(s[si]) >= 128:
			si += 1
		elif t[ti] == "<unk>":
			tii = 0
			ti += 1
			if ti >= len(t):
				break
		else:
			while s[si] != t[ti][tii]:
				si += 1
				if si >= len(s):
					break

		if tii == len(t[ti]):
			tii = 0
			ti += 1
	return caps

def scale(counter, min_=0, max_=10, smoothfactor=0):
	c_min = min(counter, key=counter.get)
	c_min = counter[c_min] - smoothfactor
	c_max = max(counter, key=counter.get)
	c_max = counter[c_max]
	new_counter = {}
	scaling = lambda x: ((max_ - min_) * (x - c_min) / (c_max - c_min + 1e-8)) + min_
	for k, v in counter.items():
		new_counter[k] = scaling(v)
	return new_counter