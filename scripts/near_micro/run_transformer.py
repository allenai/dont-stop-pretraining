import sys
from itertools import islice
import argparse
import json
import os
import torch
from pytorch_transformers import (BertModel, BertTokenizer, GPT2Model,
                                  RobertaModel, RobertaTokenizer,
                                  GPT2Tokenizer, OpenAIGPTModel,
                                  OpenAIGPTTokenizer, TransfoXLModel,
                                  TransfoXLTokenizer, XLMModel, XLMTokenizer,
                                  XLNetModel, XLNetTokenizer)
from tqdm import tqdm
from allennlp.models.archival import load_archive
MODELS = {
    "roberta": {"model": RobertaModel, "tokenizer": RobertaTokenizer, "name": 'roberta-base'},
    "bert": {"model": BertModel, "tokenizer": BertTokenizer, "name": 'bert-base-uncased'},
    "gpt": {"model": OpenAIGPTModel, "tokenizer":OpenAIGPTTokenizer, "name": 'openai-gpt'},
    "gpt2": {"model": GPT2Model,       "tokenizer": GPT2Tokenizer,   "name":   'gpt2'},
    "transxl": {"model": TransfoXLModel, "tokenizer": TransfoXLTokenizer, "name":'transfo-xl-wt103'},
    "xlnet": {"model": XLNetModel,     "tokenizer": XLNetTokenizer,   "name":  'xlnet-base-cased'},
    "xlm": {"model": XLMModel,       "tokenizer": XLMTokenizer,    "name":   'xlm-mlm-enfr-1024'}
}


def batchify_text(text_file, num_lines, batch_size):
    with open(text_file,'r') as p:
        batch_text = iter(lambda: list(islice(p, batch_size)), [])
        for lines in tqdm(batch_text,
                          total=num_lines // batch_size):
            rows = []
            ids = []
            for line in lines:
                line = json.loads(line)
                row = tokenizer.encode(line['text'], add_special_tokens=True)
                while len(row) < args.max_seq_length:
                    row.append(pad_token)
                if len(row) > args.max_seq_length:
                    row = row[:args.max_seq_length]
                rows.append(row)
                ids.append(line['index'])
            tokens = torch.tensor(rows)
            ids = torch.tensor(ids)
            yield ids, tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, required=True)
    parser.add_argument("--serialization-dir", "-s", type=str, required=True)
    parser.add_argument("--vecs-output", "-o", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--device", "-d", type=int, required=False, default=-1)
    parser.add_argument("--batch_size", "-b", type=int, required=False, default=1)
    parser.add_argument("--max_seq_length", "-l", type=int, required=False, default=40)
    args = parser.parse_args()
    pad_token = 0
    print("loading model...")
    tokenizer = MODELS[args.model]['tokenizer'].from_pretrained(MODELS[args.model]['name'])
    model = MODELS[args.model]['model'].from_pretrained(MODELS[args.model]['name'])
    args.device = args.device - 1
    if args.device > -1:
        device = torch.device("cuda", args.device)
        try:
            model.to(device)
        except RuntimeError as e:
            print(f"tried to pass {args.device} as device")
    num_lines = 0
    print("counting lines...")
    with open(args.text,'r') as p:
        for line in p:
            num_lines += 1
    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)
    vecs = []
    ids_ = []
    for ids, tokens in batchify_text(args.text, num_lines, args.batch_size):
        with torch.no_grad():
            if args.device > -1:
                try:
                    tokens = tokens.to(device)
                except RuntimeError as e:
                    print(f"tried to pass {args.device} as device")
            output = model(tokens)[0]
            doc_repr = output[:, 0, :]
            vecs.append(doc_repr)
            ids_.append(ids)
    torch.save((torch.cat(ids_, 0).type(torch.IntTensor), torch.cat(vecs, 0)).type(torch.FloatTensor),
                os.path.join(args.serialization_dir, args.vecs_output))