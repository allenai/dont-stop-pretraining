from pathlib import Path
import random
from tqdm import tqdm
import click
import argparse
from typing import List
import numpy as np
import codecs
import json

DATA_DIR = Path("/net/nfs.corp/allennlp/suching/lm_data/")

def generate_data_distribution(total_tokens: int, starve: str=None, starve_prop: float=None):
    input_dirs = [
        DATA_DIR / "1b",
        DATA_DIR / "realnews",
        DATA_DIR / "legal",
        DATA_DIR / "tweets",
        DATA_DIR / "biomed",
        DATA_DIR / "cs",
        DATA_DIR / "reviews"
    ]

    data_distribution = {}
    for ix, input_dir in enumerate(input_dirs):
        if starve:
            if input_dir.name == starve:
                proportion = (1/len(input_dirs)) * starve_prop
            else:
                leftover = 1 - ((len(input_dirs) - 1) / len(input_dirs) + (1 / len(input_dirs) * starve_prop))
                proportion = (1/len(input_dirs)) + (leftover / (len(input_dirs)-1))
        else:
            proportion = 1 / len(input_dirs)
        data_distribution[input_dir] = {
                        'domain': ix,
                        'proportion': proportion
                        }
    assert np.isclose(sum([y['proportion'] for x, y in data_distribution.items()]), 1.0)
    return data_distribution

def reservoir_sampling( file_):
    """
    reservoir sampling for reading random lines from file without loading
    entire file into memory

    See here for explanation of algorithm:
    https://stackoverflow.com/questions/35680236/select-100-random-lines-from-a-file-with-a-1-million-which-cant-be-read-into-me

    Parameters
    ----------
    file : `str` - file path
    sample_size : `int` - size of random sample you want

    Returns
    -------
    result : `List[str]` - sample lines of file
    """
    file_iterator = iter(file_)

    try:
        result = [next(file_iterator) for _ in range(sample_size)]

    except StopIteration:
        raise ValueError("Sample larger than population")

    for index, item in enumerate(file_iterator, start=sample_size):
        sample_index = np.random.randint(0, index)
        if sample_index < sample_size:
            result[sample_index] = item

    np.random.shuffle(result)

    return result


def write_jsonlist(list_of_json_objects, output_filename, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')

def sample(data_distribution, split, total_words, output_file, add_domain_token):
    res = []
    for input_dir, metadata in tqdm(data_distribution.items()):
        out = []
        count = 0
        with open(input_dir / (split + ".txt"), 'r') as f:
            while count <= int(round(total_words * metadata['proportion'])):
                out_ = {}
                line = f.readline().strip()
                count += len(line.split())
                out_['text'] = line
                out_['domain'] = metadata['domain']
                out.append(out_)
        res.extend(out)
    random.shuffle(res)
    write_jsonlist(res, output_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", '-s')
    parser.add_argument("--total_words", '-w', type=int)
    parser.add_argument("--starve", '-d', required=False, default=None)
    parser.add_argument("--starve_prop", '-p', type=float, required=False, default=0.0)
    parser.add_argument("--output_file", '-o')
    parser.add_argument("--add_domain_token", action='store_true')
    args = parser.parse_args()
    data_distribution = generate_data_distribution(total_tokens=args.total_words, starve=args.starve, starve_prop=args.starve_prop)
    sample(data_distribution, args.split, args.total_words, args.output_file, args.add_domain_token)
