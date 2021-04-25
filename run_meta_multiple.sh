gpuid=$1
updateAlgo=$2
patience=$3
taskId=$4
devWd=$5
devlr=$6
mwlr=$7
classflr=$8
metric=$9
classfdp=${10}

iterbsz=24

for k in {0..9}
do
	echo 'Running Experiment - '$k ' Update Algo '$updateAlgo 'dev weight-decay '$devWd ' dev-learning rate '$devlr ' meta-weights-lr '$mwlr ' Classifier LR = '$classflr ' Metric = '$metric ' ClassfDP = '$classfdp
	CUDA_VISIBLE_DEVICES=$gpuid python -u -m scripts.run_language_modeling_endtask --train_data_file datasets/$taskId/train.txt --line_by_line  --model_type roberta-base --tokenizer_name roberta-base --mlm --per_gpu_train_batch_size 32 --gradient_accumulation_steps 8 --model_name_or_path roberta-base --eval_data_file datasets/$taskId/dev.txt  --do_eval  --evaluate_during_training  --do_train --learning_rate 0.0001 --block_size 512 --logging_steps 5000 --dev_task_file datasets/$taskId/dev.jsonl --test_task_file datasets/$taskId/test.jsonl --classifier_dropout 0.1 --classf_lr $classflr --classf_patience $patience --num_train_epochs 150 --primary_task_id $taskId --alpha_update_algo $updateAlgo --classifier_dropout $classfdp --output_dir m4m_dsp/$taskId/$updateAlgo'_devlr'$devlr'_classfDP'$classfdp'_devWD'$devWd'_metaWLr'$mwlr'_classflr'$classflr'_config1_run'$k --overwrite_output_dir --classf_iter_batchsz  $iterbsz --classf_ft_lr 5e-6 --classf_max_seq_len 512 --seed $k  --classf_dev_wd $devWd --classf_dev_lr $devlr --meta-lr-weight $mwlr --classf-metric $metric &> static_runlogs/$taskId/$updateAlgo'_devlr'$devlr'_devWD'$devWd'_metaWLr'$mwlr'_classflr'$classflr'_config1_run'$k.txt
done

