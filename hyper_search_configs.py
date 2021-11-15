HYPER_CONFIG = {
		'auxlr': [2.0, 1.0, 5e-1],
		'soptlr': [1e-1, 5e-2],
		'classflr': [1e-4, 1e-3],
		'nconf_samp': [3, 6],
		'primbsz': [128, 256],
		'auxbsz': [256]
}


CITATION_INTENT = {
	'primtaskid': 'citation_intent',
	'trainfile':  '/home/ldery/internship/dsp/datasets/citation_intent/train.jsonl',
	'devfile':    '/home/ldery/internship/dsp/datasets/citation_intent/dev.jsonl',
	'testfile':   '/home/ldery/internship/dsp/datasets/citation_intent/test.jsonl',
	'taskdata':   '/home/ldery/internship/dsp/datasets/citation_intent/train.txt',
	'domaindata': '/home/ldery/internship/dsp/datasets/citation_intent/domain.10xTAPT.txt',
	'metric':     'f1',
}


SCIIE = {
	'primtaskid': 'sciie',
	'trainfile':  '/home/ldery/internship/dsp/datasets/sciie/train.jsonl',
	'devfile':    '/home/ldery/internship/dsp/datasets/sciie/dev.jsonl',
	'testfile':   '/home/ldery/internship/dsp/datasets/sciie/test.jsonl',
	'taskdata':   '/home/ldery/internship/dsp/datasets/sciie/train.txt',
	'domaindata': '/home/ldery/internship/dsp/datasets/sciie/domain.10xTAPT.txt',
	'metric':     'f1',
}

CHEMPROT = {
	'primtaskid': 'chemprot',
	'trainfile':  '/home/ldery/internship/dsp/datasets/chemprot/train.jsonl',
	'devfile':    '/home/ldery/internship/dsp/datasets/chemprot/dev.jsonl',
	'testfile':   '/home/ldery/internship/dsp/datasets/chemprot/test.jsonl',
	'taskdata':   '/home/ldery/internship/dsp/datasets/chemprot/train.txt',
	'domaindata': '/home/ldery/internship/dsp/datasets/chemprot/domain.10xTAPT.txt',
	'metric':     'accuracy',
}

