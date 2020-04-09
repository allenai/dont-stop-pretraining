model=$1
task=$2
dataset=$3
big=$4

if [[ $big == 1 ]];
then
    hp="ROBERTA_CLASSIFIER_BIG"
else
    hp="ROBERTA_CLASSIFIER"
fi

if [[ $model == *"med"* ]];
then
    model_sub="med"
else
    model_sub="base"
fi

if [[ $model == *"cs"* ]];
then
    model_sub="cs"
else
    model_sub="base"
fi

if [[ $model == *"review"* ]];
then
    model_sub="review"
else
    model_sub="base"
fi

if [[ $model == *"news"* ]];
then
    model_sub="news"
else
    model_sub="base"
fi


if [[ $dataset == "hyperpartisan_news" ]];
then
    srun -w allennlp-server4 -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${model_sub}_${task}_${dataset}    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --jackknife
else
    srun -w allennlp-server4 -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${model_sub}_${task}_${dataset}    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --evaluate_on_test
fi


