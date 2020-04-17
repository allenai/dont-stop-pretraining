# dont-stop-pretraining
Code associated with the Don't Stop Pretraining ACL 2020 paper


## Installation

```
conda env create -f environment.yml
conda activate domains
```

## Models

We've uploaded our DAPT and TAPT models to [huggingface](https://huggingface.co/allenai).

## Available Pretrained Models

Unlike in `master`, using the uploaded models on huggingface is easy -- no need to manually download models beforehand.

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
allenai/dsp_roberta_base_dapt_news_tapt_ag_115K
allenai/dsp_roberta_base_tapt_ag_115K
allenai/dsp_roberta_base_dapt_reviews_tapt_amazon_helpfulness_115K
allenai/dsp_roberta_base_tapt_amazon_helpfulness_115K
allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169
allenai/dsp_roberta_base_tapt_chemprot_4169
allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688
allenai/dsp_roberta_base_tapt_citation_intent_1688
allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_5015
allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_515
allenai/dsp_roberta_base_tapt_hyperpartisan_news_5015
allenai/dsp_roberta_base_tapt_hyperpartisan_news_515
allenai/dsp_roberta_base_dapt_reviews_tapt_imdb_20000
allenai/dsp_roberta_base_dapt_reviews_tapt_imdb_70000
allenai/dsp_roberta_base_tapt_imdb_20000
allenai/dsp_roberta_base_tapt_imdb_70000
allenai/dsp_roberta_base_dapt_biomed_tapt_rct_180K
allenai/dsp_roberta_base_tapt_rct_180K
allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500
allenai/dsp_roberta_base_tapt_rct_500
allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219
allenai/dsp_roberta_base_tapt_sciie_3219
```

The final numbers in each model above are the dataset sizes. Larger dataset sizes (e.g. imdb_70000 vs. imdb_20000) are curated TAPT models. These only exist for `imdb`, `rct`, and `hyperpartisan_news`.

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
        --model allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 \
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





