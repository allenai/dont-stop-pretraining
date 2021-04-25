
for k in {1..4}
do
	echo 'Running Meta-Dapt Experiment - ' $k 
	python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/domains/compsci/domain.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --per_gpu_train_batch_size 16 --gradient_accumulation_steps 18 --model_name_or_path roberta-base --eval_data_file datasets/citation_intent/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 384 --logging_steps 5000 --dev_task_file datasets/citation_intent/dev.jsonl --test_task_file datasets/citation_intent/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --classf_patience 5  --num_train_epochs 2 --primary_task_id citation_intent --alpha_update_algo meta --classifier_dropout 0.1 --output_dir m4m_dsp/citation_intent/citation_dapt/meta_mlrw=5e-2_run_$k --overwrite_output_dir --classf_iter_batchsz  15 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --seed $k  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 5e-2 --classf-metric f1 --lazy-dataset --max_steps 12500  --dev_batch_sz 32 --base_batchsz 2 &> static_runlogs/citation_intent/our_dapt_runs/meta_mlrw=5e-2_run_$k.txt
done







CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/citation_intent/domain.1xTAPT.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --eval_data_file datasets/citation_intent/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0005 --block_size 512 --logging_steps 5000 --dev_task_file datasets/citation_intent/dev.jsonl --test_task_file datasets/citation_intent/test.jsonl --classifier_dropout 0.1 --classf_lr 1e-5 --primary_task_id citation_intent --alpha_update_algo meta --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric f1 --lazy-dataset --output_dir m4m_dsp/citation_intent/small_dapt/1xTapt/meta.config1 --overwrite_output_dir --seed 0  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 1e-1  --dev_batch_sz 32 --base_batchsz 2 --classf_patience 20 --num_train_epochs 100 --classf_iter_batchsz 50 --per_gpu_train_batch_size 16 --gradient_accumulation_steps 5 --eval_every 20