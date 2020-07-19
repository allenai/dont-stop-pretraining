import argparse
import json
import os
import sys
from typing import Iterator

import torch
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from vampire.api import VampireModel


def get_json_data(input_file, predictor=None):
    if input_file == "-":
        for line in sys.stdin:
            if not line.isspace():
                if predictor:
                    yield predictor.load_line(line)
                else:
                    yield json.loads(line)
    else:
        input_file = cached_path(input_file)
        with open(input_file, "r") as file_input:
            for line in file_input:
                if not line.isspace():
                    if predictor:
                        yield predictor.load_line(line)
                    else:
                        yield json.loads(line)

def predict_json(predictor, batch_data):
    if len(batch_data) == 1:
        results = [predictor.predict_json(batch_data[0])]
    else:
        results = predictor.predict_batch_json(batch_data)
    for output in results:
        yield output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="path to vampire model archive (e.g. /path/to/model.tar.gz)"
                             " or huggingface model name (e.g. roberta-base) ")
    parser.add_argument("--output_file", type=str, required=True, help='path to output')
    parser.add_argument("--input_file", type=str, required=True, help='path to output')
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--device', type=int, required=False, default=-1)

    args = parser.parse_args()
    vectors = []
    ids = []
    if 'model.tar.gz' in args.model:
        model = VampireModel.from_pretrained(args.model, args.device, for_prediction=True)
    else:
        model = AutoModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.device >= 0:
            model = model.to(f'cuda:{args.device}')
    file_length = 0
    with open(args.input_file, 'r') as f:
        for line in f:
            file_length += 1

    if 'tar.gz' in args.model:
        predictor = model.model
    else:
        predictor = None
    file_iterator = lazy_groups_of(get_json_data(args.input_file,
                                                 predictor=predictor),
                                   args.batch_size)
    
    for batch_json in tqdm(file_iterator, total=file_length // args.batch_size):
        if 'tar.gz' in args.model:
            batch_vectors = model.extract_features(batch_json,
                                                   batch=True,
                                                   scalar_mix=True)
            for vector in batch_vectors:
                vectors.append(vector)
        else:
            lines = [x['text'] for x in batch_json]
            input_ids = tokenizer.batch_encode_plus(lines,
                                                add_special_tokens=tokenizer.add_special_tokens,
                                                truncation=True,
                                                max_length=100,
                                                pad_to_max_length=True,
                                                return_tensors='pt',
                                                padding=True)
            if args.device >= 0:
                input_ids = input_ids.to(model.device)         
            with torch.no_grad():
                out = model(**input_ids)
                vectors_ = out[0][:, 0, :]  # Models outputs are now tuples
            vectors.append(vectors_)
        indices = torch.IntTensor([x['index'] for x in batch_json]).unsqueeze(-1)
        ids.append(indices)
    torch.save((torch.cat(ids, 0).cpu(), torch.cat(vectors, 0).cpu()), args.output_file)
