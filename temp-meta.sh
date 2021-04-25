
taskid=chemprot
lr=1e-3
dp=0.3
metric='accuracy'

# mkdir -p m4m_dsp/$taskid/small_dapt/1xTapt/
# mkdir -p static_runlogs/$taskid/small_dapt/1xTapt/

# datasz=1

# for k in {0..2}
# do
# 	echo 'Running seed : '$k
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo meta --classifier_dropout $dp --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/test/$datasz'xTapt/meta.config1_'$k --overwrite_output_dir --seed $k  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 5e-2  --dev_batch_sz 32 --base_batchsz 1 --classf_patience 5 --num_train_epochs 100 --classf_iter_batchsz 8 --per_gpu_train_batch_size 9 --gradient_accumulation_steps 4 --eval_every 20 --parallelize_classifiers &> static_runlogs/$taskid/small_dapt/$datasz'xTapt/meta.config1_'$k.txt
# done



datasz=1
classfwd=1e-3
devlr=1e-3
echo 'Moving on to '$datasz'x TAPT'
# # mkdir -p static_runlogs/$taskid/small_dapt/$datasz'xTapt/'


# Remember to set the classifier iter-batchsz appropriately
for k in {1..1}
do
	echo 'Running seed : '$k
	CUDA_VISIBLE_DEVICES=2 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo meta --classifier_dropout $dp --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/$datasz'xTapt/meta.config2_'$k --overwrite_output_dir --seed $k  --classf_dev_wd $classfwd --classf_dev_lr $devlr --meta-lr-weight 5e-2  --dev_batch_sz 32 --base_batchsz 1 --classf_patience 5 --num_train_epochs 100 --classf_iter_batchsz 18 --per_gpu_train_batch_size 18 --gradient_accumulation_steps 14  --eval_every 20 &> static_runlogs/$taskid/small_dapt/$datasz'xTapt/meta.config2_'$k.txt #--parallelize_classifiers
#&> static_runlogs/$taskid/small_dapt/$datasz'xTapt/meta.config1_'$k.txt
done























# for k in {0..0}
# do
# 	echo 'Running seed : '$k
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/citation_intent/domain.1xTAPT.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/citation_intent/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/citation_intent/dev.jsonl --test_task_file datasets/citation_intent/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --primary_task_id citation_intent --alpha_update_algo meta --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric f1 --lazy-dataset --output_dir m4m_dsp/citation_intent/small_dapt/1xTapt/meta.config1_$k --overwrite_output_dir --seed $k  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 5e-2  --dev_batch_sz 32 --base_batchsz 1 --classf_patience 10 --num_train_epochs 100 --classf_iter_batchsz 64 --per_gpu_train_batch_size 12 --gradient_accumulation_steps 4 --eval_every 20 &> static_runlogs/citation_intent/small_dapt/1xTapt/meta.config1_$k.txt
# done

# echo 'Moving on to 100x TAPT'
# mkdir -p static_runlogs/citation_intent/small_dapt/100xTapt/
# mkdir -p m4m_dsp/citation_intent/small_dapt/100xTapt/

# for k in {0..2}
# do
# 	echo 'Running seed : '$k
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/citation_intent/domain.100xTAPT.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/citation_intent/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/citation_intent/dev.jsonl --test_task_file datasets/citation_intent/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --primary_task_id citation_intent --alpha_update_algo meta --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric f1 --lazy-dataset --output_dir m4m_dsp/citation_intent/small_dapt/100xTapt/meta.config1_$k --overwrite_output_dir --seed $k  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 5e-2  --dev_batch_sz 32 --base_batchsz 1 --classf_patience 10 --num_train_epochs 100 --classf_iter_batchsz 16 --per_gpu_train_batch_size 13 --gradient_accumulation_steps 16 --eval_every 20 &> static_runlogs/citation_intent/small_dapt/100xTapt/meta.config1_$k.txt
# done

# lr=1e-6
# device=0
# for k in {0..2}
# do
# 	echo 'Running seed : '$k
# 	CUDA_VISIBLE_DEVICES=$device python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/citation_intent/domain.100xTAPT.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/citation_intent/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/citation_intent/dev.jsonl --test_task_file datasets/citation_intent/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --primary_task_id citation_intent --alpha_update_algo meta --classifier_dropout 0.1 --classf_ft_lr $lr --classf_max_seq_len 512 --classf-metric f1 --lazy-dataset --output_dir m4m_dsp/citation_intent/small_dapt/100xTapt/meta.config1_$k --overwrite_output_dir --seed $k  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 5e-2  --dev_batch_sz 32 --base_batchsz 1 --classf_patience 10 --num_train_epochs 100 --classf_iter_batchsz 16 --per_gpu_train_batch_size 13 --gradient_accumulation_steps 16 --eval_every 20 --only-run-classifier &> static_runlogs/citation_intent/small_dapt/100xTapt/'meta.finetune.lr='$lr'.config1_'$k.txt
# done