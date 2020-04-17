MODEL=$1
OUTDIR=$2
URL=https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/models/$MODEL


if [[ ! -d $OUTDIR ]];
then
    mkdir -p $OUTDIR
    curl -Lo $OUTDIR/added_tokens.json ${URL}/added_tokens.json
    curl -Lo $OUTDIR/config.json ${URL}/config.json
    curl -Lo $OUTDIR/merges.txt ${URL}/merges.txt
    curl -Lo $OUTDIR/pytorch_model.bin ${URL}/pytorch_model.bin
    curl -Lo $OUTDIR/special_tokens_map.json ${URL}/special_tokens_map.json
    curl -Lo $OUTDIR/tokenizer_config.json ${URL}/tokenizer_config.json
    curl -Lo $OUTDIR/vocab.json ${URL}/vocab.json
    echo "$MODEL downloaded at path $OUTDIR."
else
    echo "$OUTDIR is not empty, aborting download..."
fi
