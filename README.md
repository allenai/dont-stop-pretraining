# dont-stop-pretraining
Code associated with the Don't Stop Pretraining ACL 2020 paper


## Installation

```
conda env create -f environment.yml
```

## Models

We've uploaded our DAPT and TAPT models to [huggingface](https://huggingface.co/allenai).

## Available Pretrained Models

Unlike in `master`, using the uploaded models on huggingface is easy -- no need to manually download models beforehand.


### DAPT models

```
allenai/cs-roberta-base
allenai/biomed-roberta-base
allenai/reviews-roberta-base
allenai/news-roberta-base
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
allenai/${DATASET}-roberta-tapt-base
allenai/${DATASET}-roberta-dapt-tapt-base
```

For `imdb`, `rct-sample`, and `hyperpartisan_news`, we additionally release Curated TAPT models:

```
allenai/${DATASET}-roberta-curated-tapt-base
allenai/${DATASET}-roberta-dapt-curated-tapt-base
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

You can supply other uploaded models to this script, just by providing the right name:

```
python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation-intent-base \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset citation_intent \
        --model allenai/citation_intent-roberta-dapt-tapt-base \
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