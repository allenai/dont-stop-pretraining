
k=$1
gpuid=$2
wconfig=$3
classfbatchsz=$4
taskid=$5
metric=$6 #'f1' # use accuracy for chemprot and f1 for sciie and citation_intent
dp=$7 # Use 0.3 for chemprot and 0.1 for the rest
ext=$8
frac=$9

lr=1e-3 # use 1e-3 for chemprot and sciie. Use 1e-5 for citation_intent
# Todo [ldery] - change this
datasz=1
nepochs=150
baselr=1e-4 # 0.0005 for dapt. Changing this back to 1e-4 for tapt related
patience=10 # Using the patience from the tapt setting

configid=$ext'.InnerConfig.'$wconfig'.classfbatchsz='$classfbatchsz
evalfreq=50
ftlr=5e-6 # old-value = 1e-6
# One thing yet to check to change the batch_size for tapt

mkdir -p m4m_dsp/$taskid/small_dapt/tapt_dapt.$datasz'x/'
mkdir -p static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/'



echo 'Running seed : '$k 'for task '$taskid
CUDA_VISIBLE_DEVICES=$gpuid python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' datasets/$taskid/train.txt --aux-task-names DAPT-MLM TAPT-MLM --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate $baselr --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo default --classifier_dropout $dp --classf_ft_lr $ftlr --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/tapt_dapt.$datasz'x/default.config'$configid'_'$k --overwrite_output_dir --seed $k --classf_patience $patience --num_train_epochs $nepochs --classf_iter_batchsz $classfbatchsz --base_batchsz 8 --per_gpu_train_batch_size 26 --gradient_accumulation_steps 10 --eval_every $evalfreq --default_weight_config $wconfig --tapt-primsize --classf_warmup_frac $frac &> static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/default.config'$configid'_'$k.txt



# echo 'Running seed : '$k 'for task '$taskid
# CUDA_VISIBLE_DEVICES=$gpuid python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' datasets/$taskid/train.txt --aux-task-names DAPT-MLM TAPT-MLM --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate $baselr --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo default --classifier_dropout $dp --classf_ft_lr $ftlr --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/tapt_dapt.$datasz'x/default.config'$configid'_'$k --overwrite_output_dir --seed $k --classf_patience $patience --num_train_epochs $nepochs --classf_iter_batchsz $classfbatchsz --base_batchsz 8 --per_gpu_train_batch_size 24 --gradient_accumulation_steps 21 --eval_every $evalfreq --default_weight_config $wconfig #&> static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/default.config'$configid'_'$k.txt

# for k in {5..9}
# do
# 	echo 'Running seed : '$k 'for task '$taskid
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' datasets/$taskid/train.txt --aux-task-names DAPT-MLM TAPT-MLM --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate $baselr --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo default --classifier_dropout $dp --classf_ft_lr $ftlr --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/tapt_dapt.$datasz'x/explore.default.config'$configid'_'$k --overwrite_output_dir --seed $k --classf_patience $patience --num_train_epochs $nepochs --classf_iter_batchsz 36 --base_batchsz 1 --per_gpu_train_batch_size 24 --gradient_accumulation_steps 7 --eval_every $evalfreq &> static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/explore.default.config'$configid'_'$k.txt
# done



