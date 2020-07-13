# Automatic Data Selection for TAPT

This README outlines the process of selecting data for TAPT using VAMPIRE (Gururangan et. al, 2019).


## Setup

Clone vampire (http://github.com/allenai/vampire) at the branch `allennlp-1.0`, and set `ROOT_DIR` and `VAMPIRE_DIR`, since we'll be switching between the directories frequently.

```bash
export ROOT_DIR=$(pwd)
git clone http://github.com/allenai/vampire
cd vampire
export VAMPIRE_DIR=$(pwd)
cd $ROOT_DIR
```

We also use GNU `parallel` in many of these commands. Install `parallel` via 

`sudo apt-get install parallel`

## Create domain and task datasets

Create datasets of domain and task examples. Make sure there is a unique id associated with each example in the datasets, in the column `index`, and a `text` field. We've included example domain and task examples on a public link:

```bash
curl -Lo domain.txt https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/examples/domain.txt
curl -Lo task.txt https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/examples/task.txt
```

If you are working with a text file that doesn't already have indices (or is not not in jsonl format), you can convert it like so:

```bash
sort domain.txt | uniq > domain.uniq
cat domain.uniq | jq  --raw-input .  | jq -rc '{"text": .}'  > domain.jsonl
jq -rc '. + {"index": input_line_number}' domain.jsonl > domain.index.jsonl 
mv domain.index.jsonl domain.jsonl


sort task.txt | uniq  > task.uniq
cat task.uniq | jq  --raw-input .  | jq -rc '{"text": .}'  > task.jsonl
jq -rc '. + {"index": input_line_number}' task.jsonl > task.index.jsonl 
mv task.index.jsonl task.jsonl
```


<!-- 
```bash
pigz -dc macro.jsonl.gz | pv | parallel --pipe -q jq -rc '.text | gsub("[\\n\\t]"; "")' | parallel --pipe -q awk 'length>3' | parallel --pipe -q jq  --raw-input .  | parallel --pipe -q jq -rc '{"text": .}' | pigz > macro.txt.noshorts.gz
pigz -dc merged.txt.gz | pv | parallel --pipe -q awk 'length>3' | parallel --pipe -q jq  --raw-input .  | parallel --pipe -q jq -rc '{"text": .}' | pigz > cs.macro.txt.noshorts.gz
zcat macro.txt.noshorts.gz | pv | sort | uniq -u | pigz > macro.txt.uniq.gz
zcat macro.txt.noshorts.gz | pv | parallel --pipe -q jq -rc '.text | gsub("[\\n\\t]"; "")' | parallel --pipe -q sort | uniq -u | pigz > macro.txt.uniq.gz
zcat macro.txt.uniq.gz | pv | perl -ne 'print if (rand() < .01)' > macro.txt
cat macro.txt | jq  --raw-input .  | jq -rc '{"text": .}'  > macro.jsonl
cat micro.txt | jq  --raw-input .  | jq -rc '{"text": .}'  > micro.jsonl
jq -rc '. + {"index": input_line_number}' train.sciie.jsonl > sciie.micro.index.jsonl
jq -rc '. + {"index": input_line_number}' cs.macro.jsonl > cs.macro.index.jsonl
mv sciie.micro.index.jsonl sciie.micro.jsonl
mv cs.macro.index.jsonl cs.macro.jsonl
``` -->

Concatenate the domain and task datasets into `world.jsonl`:

```bash
cat domain.jsonl task.jsonl | shuf > world.jsonl
```

Extract the text from `world.jsonl` using `parallel` and `jq`:

```bash
cat world.jsonl | pv | parallel --pipe -q jq -rc '.text | gsub("[\\n\\t]"; "")' > world.txt
```

## Tokenize

<!-- Train a BPE model on the world:

```bash
python scripts/tapt_selection/train_tokenizer.py --input_file world.txt --tokenizer_type BPE --serialization_dir world.bpe.model --vocab_size 5000
``` -->

Tokenize `world.jsonl`, `domain.jsonl`, and `task.jsonl` with `scispacy` (this is a biomedical domain/dataset -- check `pretokenize.py` for other tokenization options):

```bash
cat world.txt | pv | parallel --pipe -q python scripts/tapt_selection/pretokenize.py --tokenizer scispacy --lower --silent  > world.tok
cat domain.jsonl | pv | parallel --pipe -q python scripts/tapt_selection/pretokenize.py --tokenizer scispacy --json --lower --silent > domain.tok.jsonl
cat task.jsonl | pv | parallel --pipe -q python scripts/tapt_selection/pretokenize.py --tokenizer scispacy --json --lower --silent > task.tok.jsonl
```

Split world into train and dev of appropriate sizes, depending on how much you want to train VAMPIRE on.

```bash
cp world.tok world.tok.train
shuf -n 100000 world.tok > world.tok.dev
```

## Preprocess data for VAMPIRE

```bash
cd $VAMPIRE_DIR
mkdir data/
python -m scripts.preprocess_data --train-path $ROOT_DIR/world.tok.train --dev-path $ROOT_DIR/world.tok.dev --serialization-dir ${VAMPIRE_DIR}/data/world --tfidf --vocab-size 30000
```

## Train VAMPIRE on World

Train vampire on your preprocessed data, following tutorial on VAMPIRE README. You might have to reduce the learning rate and/or increase batch size if training is unstable (ie, training fails with NaN loss).

```bash
export DATA_DIR="$(pwd)/data/world"
export VOCAB_SIZE=30000 ## this value is printed after data preprocessing in previous step
export LAZY=0
python -m scripts.train --config training_config/vampire.jsonnet  --serialization-dir model_logs/vampire-world --environment VAMPIRE  --device 0  -o
```


## Extract VAMPIRE embeddings

Shard the `macro.jsonl` and `micro.jsonl` for parallel embedding extraction:

```bash
cd $ROOT_DIR
mkdir task_shards/
split --lines 100 --numeric-suffixes task.tok.jsonl task_shards/
mkdir task_emb/
mkdir domain_shards/
split --lines 100000 --numeric-suffixes domain.tok.jsonl domain_shards/
mkdir domain_emb/
```

Extract VAMPIRE embeddings on the domain and and task data using the trained VAMPIRE model from previous step.

```bash
cd $VAMPIRE_DIR
parallel --ungroup python -m scripts.run_vampire ${VAMPIRE_DIR}/model_logs/vampire-world/model.tar.gz {1} --batch 64 --include-package vampire --predictor vampire --output-file ${ROOT_DIR}/task_emb/{1/.} --silent ::: ${ROOT_DIR}/task_shards/*


# with multi-GPU setup
parallel --ungroup --jobs=8 python  -m scripts.run_vampire ${VAMPIRE_DIR}/model_logs/vampire-world/model.tar.gz {1} --batch 64 --include-package vampire --predictor vampire --output-file ${ROOT_DIR}/domain_emb/{1/.}  --cuda-device '$(expr {%} - 1)' ::: ${ROOT_DIR}/domain_shards/*

# with CPU
parallel --ungroup python -m scripts.run_vampire ${VAMPIRE_DIR}/model_logs/vampire-world/model.tar.gz {1} --batch 64 --include-package vampire --predictor vampire --output-file ${ROOT_DIR}/domain_emb/{1/.} --silent ::: ${ROOT_DIR}/domain_shards/*
```

## Run Faiss

First install faiss. If you have a GPU run 

```bash
pip install faiss-gpu
```

otherwise run

```bash
pip install faiss
```

Run FAISS k-nearest neighbors on the VAMPIRE embeddings to generate a file of near-micro examples from the macro domain.

```bash
cd ${ROOT_DIR}
python ${ROOT_DIR}/scripts/tapt_selection/convert_pytorch_to_memmap.py "task_emb/*"
python ${ROOT_DIR}/scripts/tapt_selection/convert_pytorch_to_memmap.py "domain_emb/*"

python -m scripts.tapt_selection.build_index --vecs ${ROOT_DIR}/domain_emb/ --text ${ROOT_DIR}/domain.jsonl --dim 81 --serialization_dir domain_index --index_type "Flat" --device 0 --batch-size 64

python -m scripts.tapt_selection.query_index --vecs ${ROOT_DIR}/task_emb/ --text ${ROOT_DIR}/task.jsonl --dim 81 --load-index domain_index --device 0 --batch-size 32 --k 5 --inspect > selected.knn.5
```
