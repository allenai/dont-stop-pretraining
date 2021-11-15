primtaskid='citation_intent'
trainfile='/home/ldery/internship/dsp/datasets/citation_intent/train.jsonl'
devfile='/home/ldery/internship/dsp/datasets/citation_intent/dev.jsonl'
testfile='/home/ldery/internship/dsp/datasets/citation_intent/test.jsonl'
taskdata='/home/ldery/internship/dsp/datasets/citation_intent/train.txt'
domaindata='/home/ldery/internship/dsp/datasets/citation_intent/domain.10xTAPT.txt'

gpuid=$1
pergpubsz=64
iterbsz=16
gradAccumSteps=8
lr=1e-4
classflr=1e-3
patience=10
classfdp=0.3
devlr=1e-2
iters=150
devftiters=10
devbsz=32
start=$2
end=$3
ext=$4
cfwd=0.1
temp=0.5
basewd=0.01
devWd=0.1
mstep=3
nconfigsamples=$5
auxlr=$6 #1e-1
soptlr=$7 #1e-1
wfrac=0.06
spconfig='vbasic'


for k in $(seq $start $end)
do
	echo $k
	base='nconfigsamples='$nconfigsamples'.devwd='$devWd'.basewd='$basewd'.prim-aux-lr='$soptlr'.auxiliaries-lr='$auxlr'.lr='$lr'.classflr='$classflr'.seed='$k'.iterbsz='$iterbsz'.spconfig='$spconfig'.warmupfrac='$wfrac'.devlr='$devlr'.metastepevery'$mstep'.pergpubsz='$pergpubsz'.devftiters='$devftiters'.temp'$temp'.'$ext
	echo $base
	CUDA_VISIBLE_DEVICES=$gpuid python -u -m scripts.autoaux --prim-task-id $primtaskid --train_data_file $trainfile --dev_data_file $devfile --test_data_file $testfile --output_dir autoaux_outputs/$primtaskid/$base --model_type roberta-base --model_name_or_path roberta-base  --tokenizer_name roberta-base --per_gpu_train_batch_size $pergpubsz  --gradient_accumulation_steps $gradAccumSteps --do_train --learning_rate $lr --block_size 512 --logging_steps 5000 --classf_lr $classflr --classf_patience $patience --num_train_epochs $iters  --classifier_dropout $classfdp --overwrite_output_dir --classf_iter_batchsz  $iterbsz --classf_ft_lr 1e-6 --classf_max_seq_len 512 --seed $k  --classf_dev_wd $devWd --classf_dev_lr $devlr -searchspace-config $spconfig -task-data $taskdata -in-domain-data $domaindata -num-config-samples $nconfigsamples --dev_batch_sz $devbsz --eval_every 30 -prim-aux-lr $soptlr -auxiliaries-lr $auxlr --classf_warmup_frac $wfrac --classf_wd $cfwd --base_wd $basewd --dev_fit_iters $devftiters -step-meta-every $mstep -use-factored-model -token_temp $temp --share-output-heads &> logfldrs/$base'.txt'
	#
done


