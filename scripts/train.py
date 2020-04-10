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

from environments.datasets import DATASETS
from environments.hyperparameters import HYPERPARAMETERS
from allennlp.common.params import Params


def main():
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='remove the specified serialization dir before training')
    parser.add_argument('-c', '--config', type=str, help='training config', required=True)
    parser.add_argument('-e', '--hyperparameters', type=str, help='hyperparameter environment, check environments/hyperparameters.py', required=True)
    parser.add_argument('-s', '--serialization_dir', type=Path, help='model serialization directory', required=True)
    parser.add_argument('-d', '--device', type=str, required=False, help = "cuda device to run model on.")
    parser.add_argument('-x', '--seed', nargs="+", type=int, required=False, default=[np.random.randint(0, 2**32)], help = "seed to run on. if not supplied, will choose random seed. if more than one seed supplied, will iterate.")
    parser.add_argument('--evaluate_on_test', action='store_true',  help = "if set, will evaluate on test after training")
    parser.add_argument('--dataset', type=str, help = "dataset to run on. see environments/dataset.py for dataset names.")
    parser.add_argument('--lazy',  action='store_true',  help = "if set, will read data lazily")
    parser.add_argument('--train_throttle', type=int, default=-1,  help = "if supplied, will sample training data to this many samples. Useful for debugging.")
    parser.add_argument('--skip_early_stopping', action='store_true',  help = "if set, will skip early stopping")
    parser.add_argument('--skip_training', action='store_true',  help = "if set, will skip training")
    parser.add_argument('--jackknife', action='store_true',  help = "if set, will run over jackknife samples")
    parser.add_argument('-m', '--model', type=str, required=False, default='roberta-base', help = "roberta model to use")
    parser.add_argument('--supermask_type', type=str, required=False, default='mitchell', help = "supermask type (hattie, mitchell, lhuc)")
    parser.add_argument('--pretrained_supermask_path', type=str, required=False, default="", help = "supermasks to add")
    parser.add_argument('--supermask', action='store_true',  help = "if set, will run supermask training")
    parser.add_argument('--supermask_sparsity', type=float, required=False, default=0.3, help = "set supermask sparsity level directly (0, 1)")
    parser.add_argument('--save_supermask', action='store_true',  help = "if set, will save supermasks during training")

    args = parser.parse_args()
    
    if args.device:
        os.environ['CUDA_DEVICE'] = args.device

    environment = HYPERPARAMETERS[args.hyperparameters.upper()]
    if "classifier.jsonnet" in args.config:
        dataset_type = "CLASSIFICATION"
    elif "ner.jsonnet" in args.config:
        dataset_type = "NER"
    else:
        dataset_type = "PICO"

    if not DATASETS[dataset_type].get(args.dataset):
        raise ValueError(f"{args.dataset} not a valid dataset for this config. choose from the following available datasets: {list(DATASETS[dataset_type].keys())}")
    
    os.environ['DATASET'] = args.dataset

    os.environ['PRETRAINED_SUPERMASK_PATH'] = args.pretrained_supermask_path

    os.environ['MODEL_NAME'] = args.model
    os.environ['HP_SEARCH'] = str(0)
    os.environ['DATA_DIR'] = DATASETS[dataset_type][args.dataset]['data_dir']
    
    os.environ["SUPERMASK"] = str(int(args.supermask))
    os.environ["SUPERMASK_TYPE"] = args.supermask_type.upper()
    os.environ["SAVE_SUPERMASK"] = str(int(args.save_supermask))


    for key, val in environment.items():
        os.environ[key]  = str(val)
    
    if args.supermask_sparsity is not None:
        os.environ["SPARSITY"] = str(args.supermask_sparsity)

    if "DISABLE_LAYERS" not in os.environ.keys():
        os.environ["DISABLE_LAYERS"] = "None"

    if 'CNN_HIDDEN_SIZE' not in os.environ.keys():
        os.environ['CNN_HIDDEN_SIZE'] = "64"
    
    if 'NUM_FILTERS' not in os.environ.keys():
        os.environ['NUM_FILTERS'] = "64" 
    
    if 'MAX_FILTER_SIZE' not in os.environ.keys():
        os.environ['MAX_FILTER_SIZE'] = "5"

    if 'MODEL_NAME' not in os.environ.keys():
        os.environ['MODEL_NAME'] = "roberta-base"


    os.environ['EVALUATE_ON_TEST'] = str(int(args.evaluate_on_test))
    os.environ['TRAIN_THROTTLE'] = str(int(args.train_throttle))
    os.environ['LAZY'] = str(int(args.lazy))
    os.environ['JACKKNIFE'] = str(int(args.jackknife))
    os.environ['SKIP_EARLY_STOPPING'] = str(int(args.skip_early_stopping))
    os.environ['SKIP_TRAINING'] = str(int(args.skip_training))

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
            for ext in range(0, 4):
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
            allennlp_command[-1] = str(args.serialization_dir)
            if os.path.exists(allennlp_command[-1]) and args.override:
                print(f"overriding {allennlp_command[-1]}")
                shutil.rmtree(allennlp_command[-1])
            subprocess.run(" ".join(allennlp_command), shell=True, check=True)


if __name__ == '__main__':
    main()
