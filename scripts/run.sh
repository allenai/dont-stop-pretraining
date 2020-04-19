model=$1
task=$2
dataset=$3
big=$4
name_prefix=$5
perf=$6
server=$7
num_samples=$8

if [[ $big == 1 ]];
then
    hp="ROBERTA_CLASSIFIER_BIG"
elif [[ $big == 2 ]];
then
    hp="ROBERTA_CLASSIFIER_MINI"
else
    hp="ROBERTA_CLASSIFIER_SMALL"
fi

for i in $(seq 1 $num_samples);
do 
    rand_int=$RANDOM
    srun -w $server --gpus=1 -p allennlp_hipri python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_${rand_int} \
        --model $model \
        --hyperparameters $hp \
        --perf $perf \
        --dataset $dataset \
        --device 0 \
        --override \
        --evaluate_on_test \
        --seed $rand_int
done