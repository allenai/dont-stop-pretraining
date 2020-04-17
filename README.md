# dont-stop-pretraining
Code associated with the Don't Stop Pretraining ACL 2020 paper


## Installation

```
conda env create -f environment.yml
```

## Models

We've uploaded our DAPT and TAPT models to [huggingface](https://huggingface.co/allenai).

### Working with the latest allennlp version

This repository works with a pinned allennlp version for reproducibility purposes. This pinned version of allennlp relies on `pytorch-transformers==1.2.0`, which requires you to manually download custom transformer models on disk. 

To run this code with the latest `allennlp`/ `transformers` version (and use the huggingface model repository to its full capacity) checkout the branch `latest-allennlp`. Caution that we haven't tested out all models on this branch, so your results may vary from what we report in paper.

If you'd like to use this pinned allennlp version, read on. Otherwise, checkout `latest-allennlp`.

## Available Pretrained Models


### DAPT models

The path to an available DAPT model follows the same URL structure:

```bash
https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/models/$DAPT_MODEL
```

Available values for `DAPT_MODEL`:

```
cs-roberta
med-roberta
review-roberta
news-roberta
```


### TAPT models

The path to an available model (TAPT, DAPT + TAPT, etc.) follows the same URL structure:

```bash
https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/models/$DATASET/$TAPT_MODEL
```

Available values for `DATASET`:

```
chemprot
rct-20k
rct-sample
citation_intent
sciie
amazon
imdb
ag
hyperpartisan_news
```

Available values for `TAPT_MODEL`:

```
roberta-tapt
roberta-dapt-tapt
```

For `imdb`, `rct-sample`, and `hyperpartisan_news`, we additionally release Curated TAPT models:

```
roberta-curated-tapt
roberta-dapt-curated-tapt
```

### Downloading Pretrained models

You can download a pretrained model using the `scripts/download_model.sh` script.

Just supply a dataset, model type, and output directory (in that order), like so:

```bash
bash scripts/download_tapt_model.sh \
        citation_intent \
        roberta-dapt-tapt \
        $(pwd)/pretrained_models/citation_intent/roberta-dapt-tapt
```

This will output the roberta-dapt-tapt model for Citation Intent corpus in `$(pwd)/pretrained_models/citation_intent/roberta-dapt-tapt`


Alternatively, download a DAPT model:

```bash
bash scripts/download_dapt_model.sh cs-roberta $(pwd)/pretrained_models/cs-roberta
```

## Example commands

### Run basic RoBERTa model

All task data is available on a public S3 url; check `environments/datasets.py`.

The following command will train a RoBERTa classifier on the Citation Intent corpus. Check `environments/datasets.py` for other datasets you can pass to the `--dataset` flag.

```
python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation-intent-base \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset citation_intent \
        --model roberta-base \
        --device 0 \
        --evaluate_on_test
```

You can supply other downloaded models to this script, by providing a path to the model:

```
python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation-intent-base \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset citation_intent \
        --model $(pwd)/pretrained_models/citation_intent/roberta-dapt-tapt \
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