import torch
import numpy as np
import torch.nn.functional as F
from config import Config
from torch.optim import SGD, Adam

def add_searchspace_args(parser):
	parser.add_argument('-searchopt-lr', type=float, default=1e-4)
	parser.add_argument('-num-config-samples', type=int, default=16)
	parser.add_argument('-use-factored-model', action='store_true')
	parser.add_argument('-step-meta-every', type=int, default=1)



def create_tensor(shape, init=0.0, requires_grad=True, is_cuda=True):
	inits = torch.ones(*shape) * init
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights

import pdb
class SearchOptions(object):
	def __init__(self, config, weight_lr, step_every=1, use_factored_model=True, is_cuda=True):
		self.config = config  # Store in case
		self.weight_lr = weight_lr
		self.weights = {}
		self.step_every = step_every
		self.step_counter = 0

		num_stages = self.config.num_stages()
		base_shape = [1 for _ in range(num_stages)]
		all_dims = list(base_shape)
		self.stage_order = []
		for stage_id in range(num_stages):
			stage_name, ids_ = self.config.get_stage_w_name(stage_id)
			shape = list(base_shape)
			shape[stage_id] = len(ids_)
			st_weights = create_tensor(shape, requires_grad=use_factored_model, is_cuda=is_cuda)
			self.weights[stage_name] = st_weights
			all_dims[stage_id] = shape[stage_id]
			self.stage_order.append(stage_name)
		self.weights['all'] = create_tensor(all_dims, requires_grad=True, is_cuda=is_cuda)
		self.stage_order.append('all')
		self.valid_configurations = None
		self.create_mask_for_illegal(all_dims, is_cuda)

		self.prim_weight =  create_tensor((1,), requires_grad=True, is_cuda=is_cuda)
		self.config_upstream_grad = create_tensor(all_dims, requires_grad=False, is_cuda=is_cuda)
		self.prim_upstream_grad = create_tensor((1,), requires_grad=False, is_cuda=is_cuda)


	def get_valid_configs(self):
		return self.valid_configurations
	
	def get_config_human_readable(self, config):
		name_ = ''
		for stage_id in range(len(config)):
			stage_name, ids_ = self.config.get_stage_w_name(stage_id)
			stage_ = "{}={}".format(stage_name, ids_[config[stage_id]])
			name_ = "{}|{}".format(name_, stage_) if len(name_) else stage_
		return name_
	
	def sample_configurations(self, num_samples):
		if num_samples >= len(self.valid_configurations):
			return self.valid_configurations
		idxs = np.random.choice(len(self.valid_configurations), size=num_samples, replace=False)
		return [self.valid_configurations[idx_] for idx_ in idxs]

	def create_mask_for_illegal(self, all_dims, is_cuda):
		assert len(all_dims) == 4, 'This assumes that there are only 4 stages. If not, please consider re-writing this function'
		self.valid_configurations = []
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
						else:
							self.valid_configurations.append((i, j, k, l))


	def get_weighttensor_nograd(self, softmax=True):
		with torch.no_grad():
			return self.get_weighttensor_wgrad(softmax=softmax)

	# Todo [ldery] - need to test this effectively
	def get_weighttensor_wgrad(self, softmax=True):
		this_tensor = sum([self.weights[name] for name in self.stage_order])
		shape = this_tensor.shape
		this_prim = self.prim_weight
		# Todo [ldery] - make sure to clean this up so it doesn't lead to a bug
		for k, v in self.weights.items():
			if k == 'all' or k == 'O' or k == 'mask':
				continue
			this_prim = this_prim + v[0, 0, 0, 0]

		if softmax:
			full_ = torch.cat((this_tensor.view(-1), this_prim))
			# Compute Normalization over all entries
			sm = F.softmax(full_, dim=-1)
			sm_reshaped = sm[:-1].view(*shape)
			return sm_reshaped, sm[-1]
		else:
			return this_tensor, this_prim

	# Not the cleanest way to do this but it's ok for now
	def update_grad(self, config, grad):
		if isinstance(config, tuple):
			self.config_upstream_grad[config[0], config[1], config[2], config[3]].add_(grad)
		else:
			self.prim_upstream_grad.add_(grad)
	
	def clear_grads(self):
		self.config_upstream_grad.zero_()
		self.prim_upstream_grad.zero_()

	def set_weightensor_grads(self, upstream_grad_tensor, prim_upstream_grad):
		weight_tensor, prim_weight = self.get_weighttensor_wgrad(softmax=False)
		proxy_loss = (weight_tensor * upstream_grad_tensor).sum()
		proxy_loss.backward()

		proxy_loss = (prim_weight * prim_upstream_grad).sum()
		proxy_loss.backward()

	def update_weighttensor(self):
		self.step_counter += 1
		self.set_weightensor_grads(self.config_upstream_grad, self.prim_upstream_grad)
		if (self.step_counter % self.step_every) == 0:
			num_params = self.weights['all'].numel()
			with torch.no_grad():
				for _, weight in self.weights.items():
					if not weight.requires_grad:
						continue
					if weight.grad is None:
						weight.grad = torch.zeros_like(weight)
					factor = (weight.numel() / num_params) / self.step_every
					new_weight = weight - (self.weight_lr * weight.grad * factor)
					weight.copy_(new_weight)
					weight.grad.zero_()
				# Perform updates on the primary weight
				assert self.prim_weight.grad is not None, 'Prim Weight should have gradients'
				factor = 1.0 / self.step_every
				new_prim = self.prim_weight - (self.weight_lr * self.prim_weight.grad * factor)
				self.prim_weight.copy_(new_prim)
				self.prim_weight.grad.zero_()
		self.clear_grads()


	def is_tokenlevel(self, output_id):
		return self.config.is_tokenlevel(output_id)
	
	def is_tokenlevel_lm(self, output_id):
		return self.config.is_tokenlevel_lm(output_id)

	def is_dot_prod(self, output_id):
		return self.config.is_dot_prod(output_id)

	def is_sent_classf(self, output_id):
		return self.config.is_sent_classf(output_id)
	
	def get_vocab(self, output_id):
		return self.config.get_vocab(output_id)


def run_tests():
	try:
		full_config = Config('full')
		searchOps = SearchOptions(full_config, 1.0)
		assert np.prod(searchOps.weights['all'].shape) == full_config.total_configs, 'All Tensor has wrong shape'
		assert np.prod(searchOps.weights['mask'].shape) == full_config.total_configs, 'Mask Tensor has wrong shape'
		num_valid = (searchOps.weights['mask'] == 0).sum().item()
		assert num_valid == 352, 'This is a hard coded number of valid configurations'
		upstream_grad_tensor = torch.ones_like(searchOps.weights['all']) * 0.5
		prim_upstream_grad = torch.ones_like(searchOps.prim_weight) * 0.5
		searchOps.set_weightensor_grads(upstream_grad_tensor, prim_upstream_grad)
		# Doing the gradient checking.
		have_grads = [v.grad is not None for k, v in searchOps.weights.items()][:-1]
		assert np.all(have_grads), 'All tensors except mask must have grad enabled'
		assert searchOps.prim_weight.grad is not None, 'Prim weight has no grad'
		print(searchOps.get_config_human_readable((0, 1, 1, 0)))
		msg = 'Passed.'
	except:
		msg = 'Failed.'
	print("run_tests : {}".format(msg))

if __name__ == '__main__':
	run_tests()