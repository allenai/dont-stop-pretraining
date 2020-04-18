
# Masked LM loss study

## Using HuggingFace

### Requirements 

`pip install transformers`

### Evaluate

```bash
cd huggingface_study
python mlm.py --input_file <path_to_data> --model_name_or_path <path_to_checkpoint or "roberta-base"> --mlm
```

## Using fairseq

### Requirements 

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable .

mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

pip install transformers
```

Get `https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz` and uncompress it.


### Convert a huggingface checkpoint to a fairseq checkpoint

```
cd fairseq_study
python convert_hf_to_fairseq.py  --fairseq_path <path_to_fairseq_checkpoint>  --hf_path <path_to_hf_checkpoint>
```

### Data preparation 

#### Add newlines (if there are no newlines)
`cat <path_to_file> | awk '{print $0,"\n"}' >> <path_to_new_file>`

#### Fairseq preprocessing
```bash
cd <path_to_where_fairseq_is_installed>

python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs <path_to_data> \
        --outputs <path_to_data>.bpe \
        --keep-empty \
        --workers 60

python fairseq_study/truncate.py <path_to_data>.bpe

fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --validpref <path_to_data>.truncate.bpe \
    --destdir data-bin/<data_name> \
    --workers 60
```

### Evaluate

```
cd fairseq_study
python validate_modified.py --task masked_lm --criterion masked_lm   --tokens-per-sample 512     --max-sentences 64     --log-format simple --log-interval 1     --model-overrides '{"freq_weighted_replacement": False}' --sample-break-mode complete_doc --skip-invalid-size-inputs-valid-test  --path <path_to_fairseq_checkpoint> data-bin/<data_name>
```



