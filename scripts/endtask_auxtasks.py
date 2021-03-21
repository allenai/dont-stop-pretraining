IMDB = {
	'imdb': '/home/ldery/internship/dsp/datasets/imdb_data/train.jsonl'
}

IMDB_SMALL = {
	'imdb_small': '/home/ldery/internship/dsp/datasets/imdb_data/train.small.jsonl'
}

AMAZON = {
	'amazon': '/home/ldery/internship/dsp/datasets/amazon/train.jsonl'
}

CITATION = {
	'citation_intent': '/home/ldery/internship/dsp/datasets/citation_intent/train.jsonl'
}

def get_auxtask_files(task_name):
	if task_name == 'imdb':
		return IMDB
	elif task_name == 'imdb_small':
		return IMDB_SMALL
	elif task_name == 'amazon':
		return AMAZON
	elif task_name == 'citation_intent':
		return CITATION
	else:
		raise ValueError