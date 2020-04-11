# dont-stop-pretraining
Code associated with the Don't Stop Pretraining ACL 2020 paper


## Installation

```
conda env create -f environment.yml
```

## Replicating paper results

If you would like to replicate our paper's results more closely, we recommend using the `pinned-allennlp` branch, which uses thee exact (but outdated) allennlp version that we evaluated on for the paper results.

The master branch of this repository is the updated to use the latest allennlp version, so specific performance of models may differ from what we report in paper.

## Example commands

### Run basic RoBERTa model

We have stored all the task data in `s3://allennlp/datasets/`. You can download this data into your local server with the `scripts/download_data.sh` script.

```bash
bash scripts/download.sh
```

The following command will train a RoBERTa classifier on the AG corpus. Check `scripts/train.py` for other dataset names and tasks you can pass to the `--dataset` flag.

```
python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/ag_base \
        --hyperparameters ROBERTA_CLASSIFIER \
        --dataset ag \
        --device 0 \
        --override \
        --evaluate_on_test
```

The following command will train a RoBERTa tagger on the NCBI corpus. 

```
python -m scripts.train \
        --config ./training_config/ner.jsonnet \
        --serialization_dir ./model_logs/base_ncbi \
        --hyperparameters ROBERTA_TAGGER \
        --dataset ncbi \
        --device 0 \
        --override \
        --evaluate_on_test
```

### Perform hyperparameter search

First, install `allentune`: https://github.com/allenai/allentune

Modify `search_space/classifier.jsonnet` accordingly.

Then run:
```
allentune search \
            --experiment-name ag_search \
            --num-cpus 56 \
            --num-gpus 4 \
            --search-space search_space/classifier.jsonnet \
            --num-samples 100 \
            --base-config training_config/classifier.jsonnet  \
            --include-package dont_stop_pretraining
```

Modify `--num-gpus` and `--num-samples` accordingly.