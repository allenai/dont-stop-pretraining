
taskid=sciie
lr=1e-3 # use 1e-3 for chemprot and sciie. Use 1e-5 for citation_intent
dp=0.1 # Use 0.3 for chemprot and 0.1 for the rest
metric='f1' # use accuracy for chemprot and f1 for sciie and citation_intent

# mkdir -p m4m_dsp/$taskid/small_dapt/1xTapt/
# mkdir -p static_runlogs/$taskid/small_dapt/1xTapt/

# datasz=1
# for k in {0..2}
# do
# 	echo 'Running seed : '$k 'for task '$taskid
# 	CUDA_VISIBLE_DEVICES=6,7 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo default --classifier_dropout $dp --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/$datasz'xTapt/default.config1_'$k --overwrite_output_dir --seed $k --classf_patience 5 --num_train_epochs 100 --classf_iter_batchsz 32 --base_batchsz 16 --per_gpu_train_batch_size 16 --gradient_accumulation_steps 8 --eval_every 20 &> static_runlogs/$taskid/small_dapt/$datasz'xTapt/default.config1_'$k.txt
# done


echo 'Moving on to 10x TAPT'
mkdir -p static_runlogs/$taskid/small_dapt/10xTapt/

datasz=10
for k in {3..5}
do
	echo 'Running seed : '$k 'for task '$taskid
	python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo default --classifier_dropout $dp --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/$datasz'xTapt/default.config1_'$k --overwrite_output_dir --seed $k --classf_patience 5 --num_train_epochs 100 --classf_iter_batchsz 25 --base_batchsz 1 --per_gpu_train_batch_size 8 --gradient_accumulation_steps 20 --eval_every 20 &> static_runlogs/$taskid/small_dapt/$datasz'xTapt/default.config1_'$k.txt
done


# datasz=100
# echo 'Moving on to 100x TAPT'
# mkdir -p static_runlogs/$taskid/small_dapt/$datasz'xTapt/'

# for k in {0..2}
# do
# 	echo 'Running seed : '$k
# 	CUDA_VISIBLE_DEVICES=6,7 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classifier_dropout 0.1 --classf_lr $lr --primary_task_id $taskid --alpha_update_algo default --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric f1 --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/$datasz'xTapt/default.config1_'$k --overwrite_output_dir --seed $k --classf_patience 5 --num_train_epochs 100 --classf_iter_batchsz 16 --base_batchsz 16 --per_gpu_train_batch_size 16 --gradient_accumulation_steps 32 --eval_every 20 &> static_runlogs/$taskid/small_dapt/$datasz'xTapt/default.config1_'$k.txt
# done