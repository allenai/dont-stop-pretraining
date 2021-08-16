import torch

# Note [ldery] - tested out in python console
# Might need to do more extensive tests if the need arises
class RepTransform(object):
	def __init__(self, rep_dict):
		self.rep_dict = rep_dict
	
	def get_rep_tform(self, input_shape, padding_mask, rep_tform_idx, sparsity=0.6):
		'''
			padding_mask should have 1 in indices that are not padding
		'''
		assert rep_tform_idx in  rep_dict, 'Invalid Representation Transform Index Specified {}'.format(rep_tform_idx)
		rep_name = self.rep_dict[rep_tform_idx]
		if rep_name == 'None':
			return None
		elif rep_name == 'Left-To-Right' or rep_name == 'Right-To-Left':
			left_to_right = torch.tril(torch.ones(input_shape[-1], input_shape[-1]))  # Along the sequence dimension
			right_to_left = left_to_right.T
			if rep_name == 'Left-To-Right':
				return left_to_right[None, :, :] * padding_mask[:, :, None]
			else:
				return right_to_left[None, :, :] * padding_mask[:, None, :]
		elif rep_name == 'Random-Factorized':
			# Todo [ldery] - there might be rows that are all zeros. Check if this causes an error
			rand_matrix = torch.FloatTensor(input_shape[-1], input_shape[-1]).uniform_() > (1 - sparsity)
			rand_matrix = rand_matrix * (1.0 - torch.eye(input_shape[-1]))
			return rand_matrix[None, :, :] * padding_mask[:, :, None]
		else:
			raise ValueError('Invalid Representation Transform Name given')
