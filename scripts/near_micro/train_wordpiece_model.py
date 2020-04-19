from tokenizers import SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer
import os
import json
import sys
import argparse
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=False, help="input text file, use '-' for stdin")
    parser.add_argument("--tokenizer_type", type=str, choices=['BPE', 'BBPE', "CharBPE", "BERT"], help='one of BPE, CharBPE, BBPE, BERT')
    parser.add_argument("--serialization_dir", help='path to output BPE model')
    parser.add_argument("--vocab_size", help='YTTM vocab size', type=int, default=10000)
    args = parser.parse_args()
    # Initialize a tokenizer
    
    tokenizer = {
                'BPE': SentencePieceBPETokenizer,
                "CharBPE": CharBPETokenizer,
                'BBPE': ByteLevelBPETokenizer,
                'BERT': BertWordPieceTokenizer
                }[args.tokenizer_type]

    tokenizer = tokenizer()

    # Then train it!
    tokenizer.train(args.input_file, vocab_size=args.vocab_size)
    if not os.path.isdir(args.serialization_dir):
        os.makedirs(args.serialization_dir)
    tokenizer.save(args.serialization_dir, 'tokenizer')
    with open(os.path.join(args.serialization_dir, "config.json"), "w+") as f:
        config = vars(args)
        json.dump(config, f)