import sys
import os
import argparse
import json
from tqdm import tqdm
import spacy
import scispacy
from spacy.tokenizer import Tokenizer
from tokenizers import SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer
from transformers import AutoTokenizer, AutoModel
from itertools import islice
import torch
from allennlp.common.util import lazy_groups_of, sanitize
from typing import List, Iterator, Optional
from allennlp.common.file_utils import cached_path
from vampire.models import VAMPIRE

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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fasttext", action='store_true', help='use fasttext embedding')
    parser.add_argument("--silent", action='store_true', help='if set, will silence TQDM')
    parser.add_argument("--output_file", type=str, required=True, help='path to output')
    parser.add_argument("--input_file", type=str, required=True, help='path to output')
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--device', type=int, required=False, default=-1)

    args = parser.parse_args()
    vectors = []
    ids = []
    if 'model.tar.gz' in args.model:
        model = VAMPIRE.from_pretrained(args.model, args.device)
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
        predictor = model
    else:
        predictor = None
    file_iterator = lazy_groups_of(get_json_data(args.input_file, predictor=predictor), args.batch_size)

    for batch_json in tqdm(file_iterator, total=file_length // args.batch_size):
        ids_ = [torch.IntTensor([x['index']]).unsqueeze(0) for x in batch_json]

        if 'tar.gz' in args.model:
            result = predict_json(predictor, batch_json)
            for output in result:
                vector = (torch.Tensor(output['encoder_layer_0']).unsqueeze(0)
                                        + -20 * torch.Tensor(output['encoder_layer_1']).unsqueeze(0)
                                        + torch.Tensor(output['theta']).unsqueeze(0))
                vectors.append(vector)
        else:
            lines = [x['text'] for x in batch_json]
            input_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=tokenizer.add_special_tokens, truncation=True, max_length=tokenizer.max_model_input_sizes[args.model], return_tensors='pt', padding=True)
            if args.device >= 0:
                input_ids = input_ids.to(model.device)         
            with torch.no_grad():
                out = model(**input_ids)
                vector = out[0][:, 0, :]  # Models outputs are now tuples
            vectors.append(vector)
        ids.extend(ids_)
    torch.save((torch.cat(ids,0).cpu(), torch.cat(vectors, 0).cpu()), args.output_file)

    # with open(args.input_file, 'r') as f:
    #     while True:
    #         next_n_lines = list(islice(f, args.batch_size))
    #         if not next_n_lines:
    #             break
    #         else:
                

    #             # for batch_json in tqdm():
    #             #     for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
    #             #         scalar_mix = (torch.Tensor(result['encoder_layer_0']).unsqueeze(0)
    #             #                     + -20 * torch.Tensor(result['encoder_layer_1']).unsqueeze(0)
    #             #                     + torch.Tensor(result['theta']).unsqueeze(0))
    #             #         vecs.append(scalar_mix)
    #             #         ids_.append(torch.IntTensor([model_input_json['index']]).unsqueeze(0))
    #             #         index = index + 1

    #             orig_json = [json.loads(line) for line in next_n_lines if not line.isspace()]
    #             lines = [x['text'] for x in orig_json]
    #             id = [x['index'] for x in orig_json]
    #             input_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=tokenizer.add_special_tokens, truncation=True, max_length=tokenizer.max_model_input_sizes[args.model], return_tensors='pt', padding=True)
    #             if args.device >= 0:
    #                 input_ids = input_ids.to(model.device)         
    #             with torch.no_grad():
    #                 out = model(**input_ids)
    #                 vector = out[0][:, 0, :]  # Models outputs are now tuples
    #             vectors.append(vector)
    #             ids.append(torch.IntTensor(id))
    #             pbar.update(1)
    # torch.save((torch.cat(ids,0).unsqueeze(-1).cpu(), torch.cat(vectors, 0).cpu()), args.output_file)
