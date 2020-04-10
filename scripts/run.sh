model=$1
task=$2
dataset=$3
big=$4
name_prefix=$5
server=$6

if [[ $big == 1 ]];
then
    hp="ROBERTA_CLASSIFIER_BIG"
else
    hp="ROBERTA_CLASSIFIER"
fi


if [[ $dataset == "hyperpartisan_news" ]];
then
    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_20389    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --jackknife \
        --seed 20389

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_605265    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --jackknife \
        --seed 605265

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_279507   \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --jackknife \
        --seed 279507

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_149959    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --jackknife \
        --seed 149959

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_212131    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --jackknife \
        --seed 212131
else
    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_20389    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --evaluate_on_test \
        --seed 20389

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_605265    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --evaluate_on_test \
        --seed 605265

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_279507    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --evaluate_on_test \
        --seed 279507

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/classifier.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_149959    \
        --model roberta-base \
        --hyperparameters ROBERTA_CLASSIFIER \
        --dataset $dataset \
        --device 0 \
        --override \
        --evaluate_on_test \
        --seed 149959

    srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
        --config ./training_config/$task.jsonnet \
        --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_212131    \
        --model $model \
        --hyperparameters $hp \
        --dataset $dataset \
        --device 0 \
        --override \
        --evaluate_on_test \
        --seed 212131
fi


