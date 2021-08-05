import torch
import numpy as np
import torch.nn.functional as F
from config import Config

def add_searchspace_args(parser):
	parser.add_argument('-weight-lr', type=float, default=1e-4)


def create_tensor(shape, init=0.0, requires_grad=True, is_cuda=True):
	inits = torch.ones(*shape) * init
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights

class SearchOptions(object):
	def __init__(self, config, weight_lr, is_cuda=True):
		self.config = config  # Store in case
		self.weight_lr = weight_lr
		self.weights = {}

		num_stages = self.config.num_stages()
		base_shape = [1 for _ in range(num_stages)]
		all_dims = list(base_shape)
		self.stage_order = []
		for stage_id in range(num_stages):
			stage_name, ids_ = self.config.get_stage_w_name(stage_id)
			shape = list(base_shape)
			shape[stage_id] = len(ids_)
			st_weights = create_tensor(shape, requires_grad=True, is_cuda=is_cuda)
			self.weights[stage_name] = st_weights
			all_dims[stage_id] = shape[stage_id]
			self.stage_order.append(stage_name)
		self.weights['all'] = create_tensor(all_dims, requires_grad=True, is_cuda=is_cuda)
		self.stage_order.append('all')
		self.create_mask_for_illegal(all_dims, is_cuda)

	
	def create_mask_for_illegal(self, all_dims, is_cuda):
		assert len(all_dims) == 4, 'This assumes that there are only 4 stages. If not, please consider re-writing this function'
		# This is quite inefficient but since we only do it once, I think it's ok.
		# Will try to come up with a more polished version if it raises issues
		# This also assumes that there are only 4 stages.
		self.weights['mask'] = create_tensor(all_dims, init=0.0, requires_grad=False, is_cuda=is_cuda)
		self.stage_order.append('mask')
		neg_inf = float('-inf')
		for i in range(all_dims[0]):
			for j in range(all_dims[1]):
				for k in range(all_dims[2]):
					for l in range(all_dims[3]):
						if (self.config.is_illegal((i, j, k, l))):
							self.weights['mask'][i, j, k, l] = neg_inf

	def get_weighttensor_nograd(self):
		with torch.no_grad():
			return self.get_weighttensor_wgrad()

	def get_weighttensor_wgrad(self):
		this_tensor = sum([self.weights[name] for name in self.stage_order])
		shape = this_tensor.shape
		# Compute Normalization over all entries
		sm = F.softmax(this_tensor.view(-1), dim=-1)
		sm = sm.view(*shape)
		return sm

	def set_weightensor_grads(self, upstream_grad_tensor):
		weight_tensor = self.get_weighttensor_wgrad()
		proxy_loss = (weight_tensor * upstream_grad_tensor).sum()
		proxy_loss.backward()

	def update_weighttensor(self):
		# Todo [ldery] - possibly implement exponentiated gradient descent if necessary
		with torch.no_grad():
			for _, weight in self.weights.items():
				if not weight.requires_grad:
					continue
				if weight.grad is None:
					weight.grad = torch.zeros_like(weight)
				new_weight = weight - (self.weight_lr * weight.grad)
				weight.copy_(new_weight)
				weight.grad.zero_()

def run_tests():
	try:
		full_config = Config('full')
		searchOps = SearchOptions(full_config, 1.0)
		assert np.prod(searchOps.weights['all'].shape) == full_config.total_configs, 'All Tensor has wrong shape'
		assert np.prod(searchOps.weights['mask'].shape) == full_config.total_configs, 'Mask Tensor has wrong shape'
		num_valid = (searchOps.weights['mask'] == 0).sum().item()
		assert num_valid == 352, 'This is a hard coded number of valid configurations'
		upstream_grad_tensor = torch.ones_like(searchOps.weights['all']) * 0.5
		searchOps.set_weightensor_grads(upstream_grad_tensor)
		# Doing the gradient checking.
		have_grads = [v.grad is not None for k, v in searchOps.weights.items()][:-1]
		assert np.all(have_grads), 'All tensors except mask must have grad enabled'
		msg = 'Passed.'
	except:
		msg = 'Failed.'
	print("run_tests : {}".format(msg))

if __name__ == '__main__':
	run_tests()