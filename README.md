# dont-stop-pretraining
Code associated with the Don't Stop Pretraining ACL 2020 paper


## Installation

```
conda env create -f environment.yml
```

### Working with the latest allennlp version

This repository works with a pinned allennlp version for reproducibility purposes. This pinned version of allennlp relies on `pytorch-transformers==1.2.0`, which requires you to manually download custom transformer models on disk. 

To run this code with the latest `allennlp`/ `transformers` version (and use the huggingface model repository to its full capacity) checkout the branch `latest-allennlp`. Caution that we haven't tested out all models on this branch, so your results may vary from what we report in paper.

If you'd like to use this pinned allennlp version, read on. Otherwise, checkout `latest-allennlp`.

## Available Pretrained Models

We've uploaded `DAPT` and `TAPT` models to [huggingface](https://huggingface.co/allenai).

### DAPT models

Available `DAPT` models:

```
allenai/cs_roberta_base
allenai/biomed_roberta_base
allenai/reviews_roberta_base
allenai/news_roberta_base
```


### TAPT models

Available `TAPT` models:

```
allenai/chemprot_roberta_tapt_base
allenai/chemprot_roberta_dapt_tapt_base
allenai/rct_180K_roberta_tapt_base
allenai/rct_180K_roberta_dapt_tapt_base
allenai/rct_500_roberta_tapt_base
allenai/rct_500_roberta_dapt_tapt_base
allenai/citation_intent_roberta_tapt_base
allenai/citation_intent_roberta_dapt_tapt_base
allenai/sciie_roberta_tapt_base
allenai/sciie_roberta_dapt_tapt_base
allenai/amazon_helpfulness_roberta_tapt_base
allenai/amazon_helpfulness_roberta_dapt_tapt_base
allenai/imdb_roberta_tapt_base
allenai/imdb_roberta_dapt_tapt_base
allenai/ag_roberta_tapt_base
allenai/ag_roberta_dapt_tapt_base
allenai/hyperpartisan_news_roberta_tapt_base
allenai/hyperpartisan_news_roberta_dapt_tapt_base
```

For `imdb`, `rct_500`, and `hyperpartisan_news`, we additionally release `Curated TAPT` models:

```
allenai/imdb_roberta_curated_tapt
allenai/imdb_roberta_dapt_curated_tapt
allenai/rct_500_roberta_curated_tapt
allenai/rct_500_roberta_dapt_curated_tapt
allenai/hyperpartisan_news_roberta_curated_tapt
allenai/hyperpartisan_news_roberta_dapt_curated_tapt
```

### Downloading Pretrained models

You can download a pretrained model using the `scripts/download_model.py` script.

Just supply a model type and serialization directory, like so:

```bash
python -m scripts/download_model \
        --model allenai/citation_intent_dapt_tapt_roberta_base \
        --serialization_dir $(pwd)/pretrained_models/allenai/citation_intent_dapt_tapt_roberta_base
```

This will output the `citation_intent_dapt_tapt_roberta_base` model for Citation Intent corpus in `$(pwd)/pretrained_models/allenai/citation_intent_dapt_tapt_roberta_base`

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
        --serialization_dir model_logs/citation-intent-dapt-dapt \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset citation_intent \
        --model $(pwd)/pretrained_models/allenai/citation_intent_dapt_tapt_roberta_base \
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