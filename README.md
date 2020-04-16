# dont-stop-pretraining
Code associated with the Don't Stop Pretraining ACL 2020 paper


## Installation

```
conda env create -f environment.yml
```

## Example commands

### Run basic RoBERTa model

We have stored all the task data in `s3://allennlp/datasets/`. You can download this data into your local server with the `scripts/download_data.sh` script.

```bash
bash scripts/download.sh
```

The following command will train a RoBERTa classifier on the Citation Intent corpus. Check `scripts/train.py` for other dataset names and tasks you can pass to the `--dataset` flag.

```
srun -w allennlp-server1 --gpus=1 -p allennlp_hipri python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation_intent_base \
        --hyperparameters ROBERTA_CLASSIFIER \
        --dataset citation_intent \
        --model cs_roberta_base \
        --device 0 \
        --evaluate_on_test
```

You can supply other huggingface-compatible models to the dataset, by providing a path to the model:

The following command will train a RoBERTa tagger on the NCBI corpus. 

```
python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation_intent_base \
        --hyperparameters ROBERTA_CLASSIFIER \
        --dataset citation_intent \
        --model /path/to/cs-roberta \
        --device 0 \
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