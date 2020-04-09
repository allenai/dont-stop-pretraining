dataset=$1

mkdir -p $(pwd)/datasets/${dataset}
curl -Lo $(pwd)/datasets/${dataset}/train.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/${dataset}/train.jsonl
curl -Lo $(pwd)/datasets/${dataset}/dev.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/${dataset}/dev.jsonl
curl -Lo $(pwd)/datasets/${dataset}/test.jsonl https://s3-us-west-2.amazonaws.com/allennlp/datasets/${dataset}/test.jsonl
