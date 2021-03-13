#!/bin/sh
exp=$1
dataset=$2
nruns=$3
gpuid=$4

# mkdir -p ${exp}/${dataset}/roberta-tapt
# mv ${exp}/*.* ${exp}/${dataset}/roberta-tapt
# cp special_tokens_map.json ${exp}/${dataset}/roberta-tapt
./scripts/deploy.sh ${exp} ${dataset} 0 ${nruns} ${gpuid} &> ${exp}/eval.txt
