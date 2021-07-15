
k=$1
gpuid=$2
wconfig=$3
classfbatchsz=$4
devbsz=$5
ext=$6
mlr=$7 # would like to test out 1e-1
taskid=$8
lr=1e-3 # use 1e-3 for chemprot and sciie. Use 1e-5 for citation_intent
metric=$9 #'f1' # use accuracy for chemprot and f1 for sciie and citation_intent
dp=${10} # Use 0.3 for chemprot and 0.1 for the rest

# Todo - ldery - switch this appropriately
datasz=1
nepochs=150
baselr=${11} # 0.0005 for dapt. Changing this back to 1e-4 for tapt related
frac=${12}
patience=10 # Using the patience from the tapt setting

evalfreq=50
ftlr=1e-6



mkdir -p m4m_dsp/$taskid/small_dapt/tapt_dapt.$datasz'x/'
mkdir -p static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/'

classfwd=${13}
devlr=1e-3

configid=$ext'.baselr='$baselr'.InnerConfig.'$wconfig'.classfbatchsz='$classfbatchsz'.mwlr='$mlr'.devbsz='$devbsz'.devwd='$classfwd


echo 'Running seed : '$k
CUDA_VISIBLE_DEVICES=$gpuid python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' datasets/$taskid/train.txt --aux-task-names DAPT-MLM TAPT-MLM --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate $baselr --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo meta --classifier_dropout $dp --classf_ft_lr $ftlr --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/tapt_dapt.$datasz'x/meta.config'$configid'_'$k --overwrite_output_dir --seed $k  --classf_dev_wd $classfwd --classf_dev_lr $devlr --meta-lr-weight $mlr --dev_batch_sz $devbsz --classf_patience $patience --num_train_epochs $nepochs --classf_iter_batchsz $classfbatchsz --base_batchsz 24 --per_gpu_train_batch_size 26 --gradient_accumulation_steps 10 --eval_every $evalfreq  --classf_warmup_frac $frac &> static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/meta.config'$configid'_'$k.txt



# for k in {2..2}
# do
# 	echo 'Running seed : '$k
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskid/domain.$datasz'xTAPT.txt' datasets/$taskid/train.txt --aux-task-names DAPT-MLM TAPT-MLM --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate $baselr --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classf_lr $lr --primary_task_id $taskid --alpha_update_algo meta --classifier_dropout $dp --classf_ft_lr $ftlr --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir m4m_dsp/$taskid/small_dapt/tapt_dapt.$datasz'x/meta.config'$configid'_'$k --overwrite_output_dir --seed $k  --classf_dev_wd $classfwd --classf_dev_lr $devlr --meta-lr-weight $mlr --dev_batch_sz 32 --classf_patience $patience --num_train_epochs $nepochs --classf_iter_batchsz 32 --base_batchsz 4 --per_gpu_train_batch_size 20 --gradient_accumulation_steps 8 --eval_every $evalfreq &> static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/meta.config'$configid'_'$k.txt
	
# 	#--classf_iter_batchsz 23 --per_gpu_train_batch_size 24 --gradient_accumulation_steps 11  --eval_every $evalfreq &> static_runlogs/$taskid/small_dapt/tapt_dapt.$datasz'x/meta.config'$configid'_'$k.txt #--parallelize_classifiers
# #&> static_runlogs/$taskid/small_dapt/$datasz'xTapt/meta.config1_'$k.txt
# done

# ./temp-dapt+tapt_meta.sh 0 0 0 7 64 taptbsz=prim.lrwarmup=0.06 1e-2 citation_intent f1 0.3 1e-4 0.06; ./temp-dapt+tapt_meta.sh 7 3 0 7 64 taptbsz=prim.lrwarmup=0.06 1e-2 citation_intent f1 0.3 1e-4 0.06
