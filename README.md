# dont-stop-pretraining
Code associated with the Don't Stop Pretraining ACL 2020 paper


## Citation


```bibtex
@inproceedings{dontstoppretraining2020,
 author = {Suchin Gururangan and Ana Marasović and Swabha Swayamdipta and Kyle Lo and Iz Beltagy and Doug Downey and Noah A. Smith},
 title = {Don't Stop Pretraining: Adapt Language Models to Domains and Tasks},
 year = {2020},
 booktitle = {Proceedings of ACL},
}
```

## Installation

```bash
conda env create -f environment.yml
conda activate domains
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



### Downloading Pretrained models

You can download a pretrained model using the `scripts/download_model.py` script.

Just supply a model type and serialization directory, like so:

```bash
python -m scripts.download_model \
        --model allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 \
        --serialization_dir $(pwd)/pretrained_models/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688
```

This will output the `allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688` model for Citation Intent corpus in `$(pwd)/pretrained_models/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688`


### Downloading data

All task data is available on a public S3 url; check `environments/datasets.py`.

If you run the `scripts/train.py` command (see next step), we will automatically download the relevant dataset(s) using the URLs in `environments/datasets.py`. However, if you'd like to download the data for use outside of this repository, you will have to `curl` each dataset individually:

```bash
curl -Lo train.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/chemprot/train.jsonl
curl -Lo dev.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/chemprot/dev.jsonl
curl -Lo test.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/chemprot/test.jsonl
```

## Example commands

### Run basic RoBERTa model

The following command will train a RoBERTa classifier on the Citation Intent corpus. Check `environments/datasets.py` for other datasets you can pass to the `--dataset` flag.

```bash
python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation_intent_base \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset citation_intent \
        --model roberta-base \
        --device 0 \
        --perf +f1 \
        --evaluate_on_test
```

You can supply other downloaded models to this script, by providing a path to the model:

```bash
python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation-intent-dapt-dapt \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset citation_intent \
        --model $(pwd)/pretrained_models/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 \
        --device 0 \
        --perf +f1 \
        --evaluate_on_test
```

### Perform hyperparameter search

First, install `allentune`: https://github.com/allenai/allentune

Modify `search_space/classifier.jsonnet` accordingly.

Then run:

```bash
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
