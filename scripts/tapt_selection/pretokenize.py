import sys
import os
import argparse
import json
from tqdm import tqdm
import spacy
import scispacy
from spacy.tokenizer import Tokenizer
from tokenizers import SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer
from transformers import AutoTokenizer


def load_huggingface_tokenizer(tokenizer_path: str):
    with open(os.path.join(tokenizer_path, 'config.json'), 'r') as f:
            config = json.load(f)
    tokenizer_type = config['tokenizer_type']
    tokenizer = {'BPE': SentencePieceBPETokenizer, 'BBPE': ByteLevelBPETokenizer, 'CharBPE': CharBPETokenizer, 'BERT': BertWordPieceTokenizer}[tokenizer_type]
    if tokenizer_type in ['BPE', 'BBPE']:
        vocab_file = [x for x in os.listdir(tokenizer_path) if 'vocab.json' in x][0]
        merges_file = [x for x in os.listdir(tokenizer_path) if 'merges.txt' in x][0]
        tokenizer = tokenizer(vocab_file=os.path.join(tokenizer_path, vocab_file),
                            merges_file=os.path.join(tokenizer_path, merges_file))
    else:
        vocab_file = [x for x in os.listdir(tokenizer_path) if 'vocab.txt' in x][0]
        tokenizer = tokenizer(vocab_file=os.path.join(tokenizer_path, vocab_file))
    return tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help='tokenizer type (spacy or path to pretrained tokenizer)')
    parser.add_argument("--transformer", action='store_true', help='transformer?')

    parser.add_argument("--json", action='store_true', help='is input file json?')
    parser.add_argument("--lower", action='store_true', help='lowercase?')
    parser.add_argument("--silent", action='store_true', help='if set, will silence TQDM')

    args = parser.parse_args()

    if args.transformer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    elif args.tokenizer == "spacy":
        nlp = spacy.load('en_core_web_sm')
        tokenizer = Tokenizer(nlp.vocab)
    elif args.tokenizer == 'scispacy':
        nlp = spacy.load('en_core_sci_sm')
        tokenizer = Tokenizer(nlp.vocab)
    else:
        tokenizer = load_huggingface_tokenizer(args.tokenizer)

    for line in tqdm(sys.stdin, disable=args.silent):
        if not line.isspace(): 
            if args.json:
                orig_json = json.loads(line)
                line = orig_json['text']
            if args.tokenizer in ['spacy', 'scispacy']:
                tokens = list(map(str, tokenizer(line)))
            elif args.transformer:
                tokens = tokenizer.batch_encode_plus([line], add_special_tokens=True, max_length=512)["input_ids"]
            else:
                tokens = tokenizer.encode(line).tokens
            if not args.transformer:
                line = ' '.join(tokens)
                if args.lower:
                    line = line.lower()
                
            if args.json:
                orig_json['text'] = line
                print(json.dumps(orig_json))
            elif not args.transformer:
                print(line)
            if args.transformer:
                print(json.dumps(tokens) + "\n")