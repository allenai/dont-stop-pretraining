import numpy as np
import itertools
from searchspace_options import (
	get_config_options,
	get_illegal_sets,
	get_config,
	ALL_TOKEN_OUTPUTS,
	ALL_SENT_CLASSF_OUTPUTS,
	ALL_SENT_DOT_OUTPUTS,
	ALL_TOKEN_CLASSF
)

def add_config_args(parser):
	parser.add_argument('-searchspace-config', type=str, default='basic', choices=['basic', 'vbasic', 'with-illegal', 'vbasic1', 'full'])


class Config(object):
	def __init__(
					self, config_name
				):
		self.config_name = config_name
		self.base_config = get_config(self.config_name)
		output = get_config_options(self.config_name)
		input_space, input_tform_space, rep_tform_space, out_tform_space = output
		inputs_dict = {k: v for k, v in enumerate(input_space)}
		inp_tform_dict = {k: v for k, v in enumerate(input_tform_space)}
		rep_tform_dict = {k: v for k, v in enumerate(rep_tform_space)}
		out_tform_dict = {k: v for k, v in enumerate(out_tform_space)}

		self.config = {
						'I' : inputs_dict,
						'T' : inp_tform_dict,
						'R' : rep_tform_dict,
						'O' : out_tform_dict
					}
		self.stage_map = list(self.config.keys())
		self.total_configs = np.prod([len(v) for k, v in self.config.items()])
		self.illegal_sets = get_illegal_sets(self.base_config)

	def num_stages(self):
		return len(self.config)

	def stages_dict(self):
		return self.config

	def get_stage_w_name(self, stage_idx):
		assert stage_idx < len(self.stage_map)
		return self.stage_map[stage_idx], self.config[self.stage_map[stage_idx]]

	def get_stage(self, stage_idx):
		_, stage_ = self.get_stage_w_name(stage_idx)
		return stage_

	def is_illegal(self, tuple_):
		assert len(tuple_) == 4, 'This assumes that there are only 4 stages. If not, please consider re-writing this function'
		op_list = []
		for stage_id, idx in enumerate(tuple_):
			op_name = self.get_stage(stage_id)[idx]
			op_list.append(op_name)
		op_list = set(op_list)
		for illegal in self.illegal_sets:
			if illegal.issubset(op_list):
				return True
		return False
	
	def is_tokenlevel(self, output_id):
		assert output_id in self.config['O'], 'Invalid Output ID '
		out_name = self.config['O'][output_id]
		return out_name in ALL_TOKEN_OUTPUTS

	def is_tokenlevel_lm(self, output_id):
		assert output_id in self.config['O'], 'Invalid Output ID '
		out_name = self.config['O'][output_id]
		is_tokenlevel_lm = out_name not in ALL_TOKEN_CLASSF
		if is_tokenlevel_lm:
			return None, is_tokenlevel_lm
		return [str(x) for x in range(ALL_TOKEN_CLASSF[out_name])], is_tokenlevel_lm

	def is_dot_prod(self, output_id):
		assert output_id in self.config['O'], 'Invalid Output ID '
		out_name = self.config['O'][output_id]
		return out_name in ALL_SENT_DOT_OUTPUTS

	def is_sent_classf(self, output_id):
		assert output_id in self.config['O'], 'Invalid Output ID '
		out_name = self.config['O'][output_id]
		return out_name in ALL_SENT_CLASSF_OUTPUTS.keys()

	def get_vocab(self, output_id):
		assert output_id in self.config['O'], 'Invalid Output ID '
		out_name = self.config['O'][output_id]
		return [str(x) for x in range(ALL_SENT_CLASSF_OUTPUTS[out_name])]

def run_tests():
	try:
		full_config = Config('full')
		assert full_config.total_configs == 640, 'There should be 640 configurations in the full config'
		assert full_config.num_stages() == 4, 'There should be exactly 4 stages'
		num_allowed = 640 - 288
		num_present = 0
		stages = [set((full_config.get_stage(i)).keys()) for i in range(full_config.num_stages())]
		for tuple_ in itertools.product(*stages):
			if not full_config.is_illegal(tuple_):
				num_present += 1
		assert num_allowed == num_present, 'There should be 352 valid configurations but got {}'.format(num_present)
		msg = 'Passed.'
	except:
		msg = 'Failed.'
	print("run_tests : {}".format(msg))

if __name__ == '__main__':
	run_tests()