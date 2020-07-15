TASK_DATA=$1
DOMAIN_IDX=$2

cp $TASK_DATA task.txt
sort task.txt | uniq  > task.uniq
cat task.uniq | jq  --raw-input .  | jq -rc '{"text": .}'  > task.jsonl
jq -rc '. + {"index": input_line_number}' task.jsonl > task.index.jsonl 
jq -rc '. + {"domain": 2}' task.index.jsonl  > task.index.1.jsonl 
mv task.index.1.jsonl task.jsonl

cat task.jsonl | pv | parallel --pipe -q python scripts/tapt_selection/pretokenize.py --tokenizer spacy --json --lower --silent > task.tok.jsonl
rm -rf task_shards/
mkdir task_shards/
split --lines 100 --numeric-suffixes task.tok.jsonl task_shards/
rm -rf task_emb/
mkdir task_emb/

cd $VAMPIRE_DIR
srun -w allennlp-server4 --gpus=1 -p allennlp_hipri parallel --ungroup python -m scripts.run_vampire ${VAMPIRE_DIR}/model_logs/vampire-world/model.tar.gz {1} --batch 64 --include-package vampire --predictor vampire --output-file ${ROOT_DIR}/task_emb/{1/.} --silent ::: ${ROOT_DIR}/task_shards/*


cd ${ROOT_DIR}
python ${ROOT_DIR}/scripts/tapt_selection/convert_pytorch_to_memmap.py "task_emb/*"

# srun -w allennlp-server4 --gpus=1 -p allennlp_hipri python -m scripts.tapt_selection.query_index --vecs ${ROOT_DIR}/task_emb/ --text ${ROOT_DIR}/task.jsonl --dim 81 --load-index domain_index --device 0 --batch-size 32 --k 5 --inspect > selected.knn.5

