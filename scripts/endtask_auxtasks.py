IMDB = {
	'imdb': '/home/ldery/internship/dsp/datasets/imdb_data/train.jsonl'
}

AMAZON = {
	'amazon': '/home/ldery/internship/dsp/datasets/amazon/train.jsonl'
}

def get_auxtask_files(task_name):
	if task_name == 'imdb':
		return IMDB
	elif task_name == 'amazon':
		return AMAZON
	else:
		raise ValueError