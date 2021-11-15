import argparse
import torch
import os
from hyper_search_configs import *
import math
import threading
import subprocess
import pickle as pkl
import time
import numpy as np
from itertools import product

# Code burrowed from : https://stackoverflow.com/questions/984941/python-subprocess-popen-from-a-thread
class MyThreadClass(threading.Thread):
	def __init__(self, command_list):
		self.stdout = None
		self.stderr = None
		self.command_list = command_list
		threading.Thread.__init__(self)

	def run(self):
		for cmd in self.command_list:
			os.system(cmd)



def add_hyperparam_options(parser):
	parser.add_argument('-task', type=str, default='sciie')
	parser.add_argument('-base-spconfig', type=str, default='vbasic1')
	parser.add_argument('-warmup-frac', type=float, default=0.06)
	parser.add_argument('-step-meta-every', type=int, default=3)
	parser.add_argument('-patience', type=int, default=10)
	parser.add_argument('-classfdp', type=float, default=0.3)
	parser.add_argument('-iters', type=int, default=150)
	parser.add_argument('-dev-ft-iters', type=int, default=10)
	parser.add_argument('-devbsz', type=int, default=32)
	parser.add_argument('-devlr', type=float, default=1e-2)
	parser.add_argument('-tokentform-temp', type=float, default=0.5)
	parser.add_argument('-base-wd', type=float, default=0.01)
	parser.add_argument('-dev-wd', type=float, default=0.1)
	parser.add_argument('-classf-wd', type=float, default=0.1)
	parser.add_argument('-lr', type=float, default=1e-4)
	parser.add_argument('-grad-accum-steps', type=int, default=2)
	parser.add_argument('-num-seeds', type=int, default=3)
	parser.add_argument('-output-dir', type=str, default='autoaux_outputs')
	parser.add_argument('-logdir', type=str, default='HyperParamLogs')
	parser.add_argument('-runthreads', action='store_true')


def get_task_info(args):
	if args.task == 'citation_intent':
		return CITATION_INTENT
	elif args.task == 'sciie':
		return SCIIE
	elif args.task == 'chemprot':
		return CHEMPROT

def get_all_hyperconfigs(config_dict):
	all_hypers = []
	all_configs = list(product(*list(config_dict.values())))
	all_keys = list(config_dict.keys())
	all_hyperconfigs = [{k: v for k, v in zip(all_keys, config)} for config in all_configs]
	return all_hyperconfigs

def get_base_runstring(args, gpuid, config, task_info):
	hyper_id = ".".join(["{}={}".format(k, v) for k, v in config.items()])
	pergpubsz = int(config['auxbsz'] / args.grad_accum_steps)
	primiterbsz = int(config['primbsz'] / args.grad_accum_steps)

	logdir = "{}/MassLaunch/{}/{}".format(args.logdir, args.task, hyper_id)
	os.makedirs(logdir, exist_ok=True)

	run_commands, outdirs = [], []
	for seed in range(args.num_seeds):
		logfile = "{}/seed={}.txt".format(logdir, seed)
		outputdir ="{}/{}/{}/seed={}".format(args.output_dir, args.task, hyper_id, seed)
		os.makedirs(outputdir, exist_ok=True)
		
		run_command = "CUDA_VISIBLE_DEVICES={} python -u -m scripts.autoaux --prim-task-id {} --train_data_file {} --dev_data_file {} --test_data_file {} --output_dir {} --model_type roberta-base --model_name_or_path roberta-base  --tokenizer_name roberta-base --per_gpu_train_batch_size {}  --gradient_accumulation_steps {} --do_train --learning_rate {} --block_size 512 --logging_steps 10000 --classf_lr {} --classf_patience {} --num_train_epochs {} --classifier_dropout {} --overwrite_output_dir --classf_iter_batchsz  {} --classf_ft_lr 1e-6 --classf_max_seq_len 512 --seed {}  --classf_dev_wd {} --classf_dev_lr {} -searchspace-config {} -task-data {} -in-domain-data {} -num-config-samples {} --dev_batch_sz {} --eval_every 30 -prim-aux-lr {} -auxiliaries-lr {} --classf_warmup_frac {} --classf_wd {} --base_wd {} --dev_fit_iters {} -step-meta-every {} -use-factored-model -token_temp {} --share-output-heads --classf-metric {} &> {}".format(gpuid, args.task, task_info['trainfile'], task_info['devfile'], task_info['testfile'], outputdir, pergpubsz, args.grad_accum_steps, args.lr, config['classflr'], args.patience, args.iters, args.classfdp, primiterbsz, seed, args.dev_wd, args.devlr, args.base_spconfig, task_info['taskdata'], task_info['domaindata'], config['nconf_samp'], args.devbsz, config['soptlr'], config['auxlr'], args.warmup_frac, args.classf_wd, args.base_wd, args.dev_ft_iters, args.step_meta_every, args.tokentform_temp, task_info['metric'], logfile)
		run_commands.append(run_command)
		outdirs.append(outputdir)
	
	return hyper_id, run_commands, outdirs


import pdb
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	add_hyperparam_options(parser)
	args = parser.parse_args()
	task_info = get_task_info(args)
	all_hyperconfigs = get_all_hyperconfigs(HYPER_CONFIG)
	num_gpus = torch.cuda.device_count() + 1
	cmnd_bsz = math.ceil(len(all_hyperconfigs) / num_gpus)
	all_threads = []
	all_conf_results = {} 
	print('This is run threads bool : ',args.runthreads)
	print('Generating configs and sending them to threads')
	for gpuid in range(num_gpus):
		hyperconfigs = all_hyperconfigs[cmnd_bsz * gpuid : (gpuid + 1)*cmnd_bsz]
		this_commands = []
		for config_ in hyperconfigs:
			hyper_id, run_commands, outdirs = get_base_runstring(args, gpuid, config_, task_info)
			this_commands.extend(run_commands)
			all_conf_results[hyper_id] = outdirs, config_
		if args.runthreads:
			this_thread = MyThreadClass(this_commands)
			all_threads.append(this_thread)
			this_thread.start()

	if args.runthreads:
		for thread in all_threads:
			thread.join()

	# We can now write the results to a csv
	print('All threads are done. Gather the configs and generate csv of results')
	timestr = time.strftime("%Y%m%d-%H%M%S")
	fname = "resultsSheets/{}.csv".format(timestr)
	with open(fname, 'w') as fhandle:
		headnames = list(HYPER_CONFIG.keys())
		headnames.extend(['preft.f1.mean', 'preft.f1.std', 'postft.f1.mean', 'postft.f1.std'])
		headnames.extend(['preft.acc.mean', 'preft.acc.std', 'postft.acc.mean', 'postft.acc.std',])
		header = ",".join(headnames)
		fhandle.write("{}\n".format(header))
		# Now we just gather the results
		for hyper_id, (outdirs, config_) in all_conf_results.items():
			preft_f1, preft_acc, postft_f1, postft_acc = [], [], [], []
			for outdir in outdirs:
				try:
					handle = open('{}/ftmodel.bestperfs.pkl'.format(outdir), 'rb')
					info = pkl.load(handle)
					pre_ft, post_ft = info
					preft_f1.append(pre_ft['f1'])
					postft_f1.append(post_ft['f1'])
					preft_acc.append(pre_ft['accuracy'])
					postft_acc.append(post_ft['accuracy'])
				except:
					print('Could not load results. Had to skip : {}'.format(outdir))

			this_results = [config_[k] for k in list(HYPER_CONFIG.keys())]
			this_results.extend([np.mean(preft_f1), np.std(preft_f1), np.mean(postft_f1), np.std(postft_f1)])
			this_results.extend([np.mean(preft_acc), np.std(preft_acc), np.mean(postft_acc), np.std(postft_acc)])
			this_entry = ",".join([str(x) for x in this_results])
			fhandle.write("{}\n".format(this_entry))

