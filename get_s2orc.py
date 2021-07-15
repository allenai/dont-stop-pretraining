"""


Example of how one would download & process a single batch of S2ORC to filter to specific field of study.
Can be useful for those who can't store the full dataset onto disk easily.
Please adapt this to your own field of study.


Creates directory structure:

|-- metadata/
	|-- raw/
		|-- metadata_0.jsonl.gz      << input; deleted after processed
	|-- medicine/
		|-- metadata_0.jsonl         << output
|-- pdf_parses/
	|-- raw/
		|-- pdf_parses_0.jsonl.gz    << input; deleted after processed
	|-- medicine/
		|-- pdf_parses_0.jsonl       << output

"""


import os
import subprocess
import gzip
import io
import json
import sys
import pdb
from tqdm import tqdm

MAX = 2.7E6

# process single batch
def process_batch(field:list, batch: dict):
	# first, let's filter metadata JSONL to only papers with a particular field of study.
	# we also want to remember which paper IDs to keep, so that we can get their full text later.
	paper_ids_to_keep = set()
	with gzip.open(batch['input_metadata_path'], 'rb') as gz, open(batch['output_metadata_path'], 'wb') as f_out:
		f = io.BufferedReader(gz)
		for line in tqdm(f.readlines()):
			metadata_dict = json.loads(line)
			paper_id = metadata_dict['paper_id']
			mag_field_of_study = metadata_dict['mag_field_of_study']
			has_body = False
			if metadata_dict['has_pdf_parse']:
				has_body = metadata_dict['has_pdf_parsed_body_text']
			field_present = False
			if mag_field_of_study is not None:
				for f_ in field:
					if (f_ in mag_field_of_study):
						field_present = True
						break

			if mag_field_of_study and field_present and has_body:     # TODO: <<< change this to your filter
				paper_ids_to_keep.add(paper_id)
				f_out.write(line)

	# now, we get those papers' full text
	with gzip.open(batch['input_pdf_parses_path'], 'rb') as gz, open(batch['output_pdf_parses_path'], 'w') as f_out:
		f = io.BufferedReader(gz)
		for line in tqdm(f.readlines()):
			metadata_dict = json.loads(line)
			paper_id = metadata_dict['paper_id']
			if paper_id in paper_ids_to_keep:
				for section in metadata_dict['body_text']:
					f_out.write('{}\n'.format(section['text']))
	return len(paper_ids_to_keep)


if __name__ == '__main__':

	field = sys.argv[1]
	METADATA_OUTPUT_DIR = os.path.join(sys.argv[2], field, 'metadata')
	PDF_PARSES_OUTPUT_DIR = os.path.join(sys.argv[2], field, 'pdfs')
	field = field.split('-')
	print('These are the fields : ', field)

	os.makedirs(METADATA_OUTPUT_DIR, exist_ok=True)
	os.makedirs(PDF_PARSES_OUTPUT_DIR, exist_ok=True)

	# TODO: make sure to put the links we sent to you here
	# there are 100 shards with IDs 0 to 99. make sure these are paired correctly.
	meta_path = "/projects/tir1/corpora/s2orc/20200705v1/full/metadata/metadata_{}.jsonl.gz"
	pdf_path = "/projects/tir1/corpora/s2orc/20200705v1/full/pdf_parses/pdf_parses_{}.jsonl.gz"
	download_linkss = [
		{"metadata": meta_path.format(i), "pdf_parses": pdf_path.format(i)} for i in range(100)
	]

	# turn these into batches of work
	# TODO: feel free to come up with your own naming convention for 'input_{metadata|pdf_parses}_path'
	batches = [{
		'input_metadata_path': download_links['metadata'],
		'output_metadata_path': os.path.join(METADATA_OUTPUT_DIR,
											 os.path.basename(download_links['metadata'].split('.gz')[0])),
		'input_pdf_parses_path': download_links['pdf_parses'],
		'output_pdf_parses_path': os.path.join(PDF_PARSES_OUTPUT_DIR,
											   os.path.basename(download_links['pdf_parses'].split('.gz')[0])),
	} for download_links in download_linkss]

	batches = tqdm(batches, total=len(batches), desc="Unzip Data")
	total = 0
	for idx, batch in enumerate(batches):
		n_processed = process_batch(field=field, batch=batch)
		total += n_processed
		if total > MAX:
			break
		print('This is the current total : {}'.format(total))
