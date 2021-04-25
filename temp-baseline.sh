echo 'Running Baselines. DAPT and DAPT->TAPT'

# Yet to run chemprot
taskid=sciie
taptfile=datasets/$taskid/train.txt
metric='accuracy'

datasz=(10)
for k in "${datasz[@]}"
do
	datafile='datasets/'$taskid'/domain.'$k'xTAPT.txt'
	savefldr='/home/ldery/internship/dsp/m4m_dsp/'$taskid'/small_dapt/'$k'xTapt/'
	logfldr='static_runlogs/'$taskid'/small_dapt/'$k'xTapt'
	echo 'Performing DAPT on data_sz '$k
	python -u -m scripts.run_language_modeling_endtask --train_data_file $datafile --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --primary_task_id $taskid --alpha_update_algo alt --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir $savefldr/dapt_$k --overwrite_output_dir --seed $k --classf_patience 1 --num_train_epochs 100 --classf_iter_batchsz 8 --per_gpu_train_batch_size 6 --gradient_accumulation_steps 42 --eval_every 5000 --init-val 1.0 --prim-start 1.01 --no_final_finetuning &> $logfldr/dapt_$k.txt
	echo 'Done with DAPT. Now PERFORMING TAPT '$K
	## NEED TO COPY THE DATA OVER HERE
	mkdir -p $savefldr/roberta-dapt-tapt_$k/checkpoint-0/
	cp $savefldr'dapt_'$k'/'* $savefldr'roberta-dapt-tapt_'$k/checkpoint-0/

	python -u -m scripts.run_language_modeling_endtask --train_data_file $taptfile --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/$taskid/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskid/dev.jsonl --test_task_file datasets/$taskid/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --primary_task_id $taskid --alpha_update_algo alt --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir $savefldr/roberta-dapt-tapt_$k --overwrite_output_dir --seed $k --classf_patience 100 --num_train_epochs 100 --classf_iter_batchsz 8 --per_gpu_train_batch_size 8 --gradient_accumulation_steps 32 --eval_every 5000 --init-val 1.0 --prim-start 1.01 --should_continue --no_final_finetuning &> $logfldr/dapt_to_tapt_$k.txt

done
