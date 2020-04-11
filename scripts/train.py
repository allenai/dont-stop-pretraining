import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from typing import Any, Dict
from pathlib import Path
import numpy as np

from allennlp.common.params import Params

DATASETS = {
    "hatespeech": {
        "data_dir": "s3://suching-dev/textcat/twitter/hatespeech/",
    },
    "ag": {
        "data_dir": "s3://suching-dev/textcat/news/ag/",
    },
    "scicite": {
        "data_dir": "s3://suching-dev/textcat/science/sci-cite/",
    },
    "citation_intent": {
        "data_dir": "s3://suching-dev/textcat/science/citation_intent/",
    },
    "chemprot": {
        "data_dir": "s3://suching-dev/textcat/science/chemprot/",
    },
    "sciie": {
        "data_dir": "s3://suching-dev/textcat/science/sciie/",
    },
    "hyperpartisan_news": {
        "data_dir": "s3://suching-dev/textcat/news/hyperpartisan_by_article/",
    },
    "biased_news": {
        "data_dir": "s3://suching-dev/textcat/news/biased_news/",
    },
    "imdb": {
        "data_dir": "s3://suching-dev/textcat/reviews/imdb/",
    },
    "amazon": {
        "data_dir": "s3://suching-dev/textcat/reviews/amazon/",
    },
    "yelp": {
        "data_dir": "s3://suching-dev/textcat/reviews/yelp/",
    },
    "twitter_sentiment": {
        "data_dir": "s3://suching-dev/textcat/twitter/semeval_2017_ task_4A/",
    },
    "twitter_irony_task_a": {
        "data_dir": "s3://suching-dev/textcat/twitter/semeval_2018_task3_irony_detection/task_a/",
    },
    "twitter_irony_task_b": {
        "data_dir": "s3://suching-dev/textcat/twitter/semeval_2018_task3_irony_detection/task_b/",
    },
    "rct-20k": {
        "data_dir": "s3://suching-dev/textcat/science/rct-sample/",
    },
    "cs-abstruct": {
        "data_dir": "s3://suching-dev/textcat/science/csabstruct-reformat/",
    }
}

random_int = random.randint(0, 2**32)

def main():
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='remove the specified serialization dir before training')
    parser.add_argument('-c', '--config', type=str, help='training config', required=True)
    parser.add_argument('-s', '--serialization-dir', type=Path, help='model serialization directory', required=True)
    parser.add_argument('-d', '--device', type=str, required=False, help = "cuda device to run model on.")
    parser.add_argument('-x', '--seed', nargs="+", type=int, required=False, default=[np.random.randint(0, 1000000)], help = "seed to run on. if not supplied, will choose random seed. if more than one seed supplied, will iterate.")
    parser.add_argument('--learning_rate', type=float, help = "roberta clf learning rate", default=8e-6)
    parser.add_argument('--dropout', type=float, help = "roberta clf dropout",  default=0.1)
    parser.add_argument('--evaluate_on_test', action='store_true',  help = "if set, will evaluate on test after training")
    parser.add_argument('--dataset', type=str, help = "dataset to run on. see environments/dataset.py for dataset names.")
    parser.add_argument('-m', '--model_name', type=Path, help = "roberta model to run. set to roberta-base or path to fine-tuned model.")
    parser.add_argument('--lazy',  action='store_true',  help = "if set, will read data lazily")
    parser.add_argument('--train_throttle', type=int, default=-1,  help = "if supplied, will sample training data to this many samples. Useful for debugging.")
    parser.add_argument('--dev_throttle', type=int, default=0,  help = "if supplied, will sample dev data to this many samples. Useful for debugging.")
    parser.add_argument('--num_epochs', type=int, default=10,  help = "how many epochs to train.")
    parser.add_argument('--grad_acc', type=int, required=False, help = "how large is gradent acc batch size.")
    parser.add_argument('--batch_size', '-b', type=int, default=4,  help = "how large is the ebatch size.")
    parser.add_argument('--skip_early_stopping', action='store_true',  help = "if set, will skip early stopping")
    parser.add_argument('--jackknife', action='store_true',  help = "if set, will run over jackknife samples")

    args = parser.parse_args()
    
    if args.device:
        os.environ['CUDA_DEVICE'] = args.device


    if not DATASETS.get(args.dataset):
        raise ValueError(f"{args.dataset} not a valid dataset. choose from the following available datasets: {list(DATASETS.keys())}")
    os.environ['DATASET'] = args.dataset
    os.environ['DATA_DIR'] = DATASETS[args.dataset]['data_dir']
    os.environ['EVALUATE_ON_TEST'] = str(int(args.evaluate_on_test))
    os.environ['LEARNING_RATE'] = str(args.learning_rate)
    os.environ['DROPOUT'] = str(args.dropout)
    os.environ['LAZY'] = str(int(args.lazy))
    os.environ['TRAIN_THROTTLE'] = str(int(args.train_throttle))
    os.environ['DEV_SAMPLE'] = str(int(args.dev_throttle))
    os.environ['NUM_EPOCHS'] = str(int(args.num_epochs))
    os.environ['BATCH_SIZE'] = str(int(args.batch_size))
    os.environ['JACKKNIFE'] = str(int(args.jackknife))

    if args.grad_acc:
        os.environ['GRAD_ACC_BATCH_SIZE'] = str(int(args.grad_acc))
    else:
        os.environ['GRAD_ACC_BATCH_SIZE'] = str(int(args.batch_size))
    os.environ['SKIP_EARLY_STOPPING'] = str(int(args.skip_early_stopping))


    if args.model_name:
        os.environ['MODEL_NAME'] = str(args.model_name)

    
    allennlp_command = [
            "allennlp",
            "train",
            "--include-package",
            "dont_stop_pretraining",
            args.config,
            "-s",
            str(args.serialization_dir)
    ]
    for seed in args.seed:
        
        os.environ['SEED'] = str(seed)  
        if args.jackknife:
            for ext in range(0, 100):
                allennlp_command[-1] = str(args.serialization_dir) + "_" + str(seed)
                os.environ['JACKKNIFE_EXT'] = str(ext)
                allennlp_command[-1] = allennlp_command[-1] + "_" + str(ext)
                if os.path.exists(allennlp_command[-1]) and args.override:
                    print(f"overriding {allennlp_command[-1]}")
                    shutil.rmtree(allennlp_command[-1]) 
                try:
                    subprocess.run(" ".join(allennlp_command), shell=True, check=True)
                except:
                    break
        else:
            allennlp_command[-1] = str(args.serialization_dir) + "_" + str(seed)
            if os.path.exists(allennlp_command[-1]) and args.override:
                print(f"overriding {allennlp_command[-1]}")
                shutil.rmtree(allennlp_command[-1])
            subprocess.run(" ".join(allennlp_command), shell=True, check=True)


if __name__ == '__main__':
    main()
