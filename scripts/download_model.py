from transformers import AutoTokenizer, AutoModel
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        help='model to download',
                        required=True)
    parser.add_argument('-s',
                        '--serialization_dir',
                        type=str,
                        help='serialization directory',
                        required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    if not os.path.isdir(args.serialization_dir):
        os.makedirs(args.serialization_dir)
    tokenizer.save_pretrained(args.serialization_dir)
    model.save_pretrained(args.serialization_dir)