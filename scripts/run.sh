model=$1
task=$2
dataset=$3
big=$4
name_prefix=$5
perf=$6
task_type=$7
server=$8

if [[ $task == 1 ]];
then
    task_type="CLASSIFIER"
else
    task_type="NER"
fi


if [[ $big == 1 ]];
then
    hp="ROBERTA_${task_type}_BIG"
elif [[ $big == 2 ]];
then
    hp="ROBERTA_${task_type}_MINI"
else
    hp="ROBERTA_${task_type}_SMALL"
fi

srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
    --config ./training_config/$task.jsonnet \
    --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_20389    \
    --model $model \
    --hyperparameters $hp \
    --perf $perf \
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
    --perf $perf \
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
    --perf $perf \
    --dataset $dataset \
    --device 0 \
    --override \
    --evaluate_on_test \
    --seed 279507

srun -w $server -p allennlp_hipri --gpus=1 python -m scripts.train \
    --config ./training_config/classifier.jsonnet \
    --serialization_dir ./model_logs/${name_prefix}_${task}_${dataset}_149959    \
    --model $model \
    --hyperparameters $hp \
    --perf $perf \
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
    --perf $perf \
    --dataset $dataset \
    --device 0 \
    --override \
    --evaluate_on_test \
    --seed 212131
