gpuid=$1
altfreq=$2
updateAlgo=$3
initVal=$4
patience=$5
primstart=$6
taskId=$7

for k in {6..10}
do
	echo 'Running Experiment - '$k ' for altfreq = '$altfreq ' with Update Algo = '$updateAlgo ' prim-start = '$primstart
	CUDA_VISIBLE_DEVICES=$gpuid python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskId/train.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --per_gpu_train_batch_size 32 --gradient_accumulation_steps 8 --model_name_or_path roberta-base --eval_data_file datasets/$taskId/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0001 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskId/dev.jsonl --test_task_file datasets/$taskId/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --classf_patience $patience --num_train_epochs 150 --primary_task_id $taskId --alpha_update_algo $updateAlgo --output_dir m4m_dsp/$taskId/$updateAlgo'_'$altfreq'_config1_run'$k --alt-freq $altfreq --overwrite_output_dir --classf_iter_batchsz  32 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --seed $k --init-val $initVal --prim-start $primstart &> static_runlogs/$taskId/$updateAlgo'_'$altfreq'_config1_run'$k.txt
done