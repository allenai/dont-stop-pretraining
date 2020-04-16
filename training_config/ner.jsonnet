local MODEL_NAME = std.extVar("MODEL_NAME");
// data directory
local DATA_DIR = std.extVar("DATA_DIR");
// whether or not to evaluate on test
local EVALUATE_ON_TEST = std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1; 
// size of dataset
local DATASET_SIZE = std.parseInt(std.extVar("DATASET_SIZE"));
// learning rate
local LEARNING_RATE = std.extVar("LEARNING_RATE");
// dropout
local DROPOUT = std.extVar("DROPOUT");
// seed
local SEED = std.parseInt(std.extVar("SEED"));
// number of epochs
local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));
// lazy mode
local LAZY = std.parseInt(std.extVar("LAZY")) == 1;
// batch size
local BATCH_SIZE = std.parseInt(std.extVar("BATCH_SIZE"));
// will sample this amount of training data, if set
local TRAIN_THROTTLE = std.parseInt(std.extVar("TRAIN_THROTTLE"));
// gradient accumulation batch size
local GRAD_ACC = std.parseInt(std.extVar("GRAD_ACC_BATCH_SIZE"));
// skip early stopping? turning this on will prevent dev eval at each epoch.
local SKIP_EARLY_STOPPING = std.parseInt(std.extVar("SKIP_EARLY_STOPPING")) == 1;

local LR_SCHEDULE = std.parseInt(std.extVar("LR_SCHEDULE")) == 1;
// validation metric
local VALIDATION_METRIC = std.extVar("VALIDATION_METRIC");
// are we jackknifing? only for hyperpartisan.
local JACKKNIFE = std.parseInt(std.extVar("JACKKNIFE")) == 1;
// jacknife file extension. only for hyperpartisan.
local JACKKNIFE_EXT = std.extVar("JACKKNIFE_EXT");
// embedding to use
local EMBEDDING = std.extVar("EMBEDDING");
// width multiplier on hidden size of feedforward network
local FEEDFORWARD_WIDTH_MULTIPLIER = std.parseInt(std.extVar("FEEDFORWARD_WIDTH_MULTIPLIER"));
// number of feedforward layers
local NUM_FEEDFORWARD_LAYERS = std.parseInt(std.extVar("NUM_FEEDFORWARD_LAYERS"));
// early stopping patience
local PATIENCE = std.parseInt(std.extVar("PATIENCE"));

local TRAIN_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "train." + JACKKNIFE_EXT else DATA_DIR  + "train.txt";
local DEV_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "dev." + JACKKNIFE_EXT else DATA_DIR  + "dev.txt";
local TEST_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "dev." + JACKKNIFE_EXT else DATA_DIR  + "test.txt";

// ----------------------------
// INPUT EMBEDDING
// ----------------------------
local PRETRAINED_ROBERTA_FIELDS(TRAINABLE) = {
  "indexer": {
    "roberta": {
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME,
        "do_lowercase": false
    }
  },
  "embedder": {
    "roberta": {
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME
    }
  },
  "tokenizer": {
    "type": "pretrained_transformer",
    "model_name": MODEL_NAME,
    "do_lowercase": false,
    "start_tokens": ["<s>"],
    "end_tokens": ["</s>"]
    },
  "optimizer": {
    "type": "bert_adam",
    "b1": 0.9,
    "b2": 0.98,
    "e": 1e-06,
    "lr": LEARNING_RATE,
    "max_grad_norm": 1,
    "parameter_groups": [
        [
            [
                "bias",
                "LayerNorm.bias",
                "LayerNorm.weight",
                "layer_norm.weight"
            ],
            {
                "weight_decay": 0
            },
            []
        ]
    ],
    "schedule": "warmup_linear",
    "t_total": if LR_SCHEDULE then (DATASET_SIZE / GRAD_ACC) * NUM_EPOCHS else -1,
    "warmup": 0.06,
    "weight_decay": 0.1
   },
  "scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
  "checkpointer": {
    "type": "fine-tuning",
    "num_epochs": NUM_EPOCHS,
    "num_serialized_models_to_keep": 0
  },
  "embedding_dim": 768
};


// CLS pooler fields
local CLS_FIELDS(embedding_dim) = {
    "type": "cls_pooler",
    "embedding_dim": embedding_dim
};

local ROBERTA_TRAINABLE = true;

local ROBERTA_EMBEDDING_DIM = PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['embedding_dim'];
local ENCODER_OUTPUT_DIM = PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['embedding_dim'];

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED, 
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": PRETRAINED_ROBERTA_FIELDS(true)['indexer'],
  },
  "train_data_path": TRAIN_DATA_PATH,
  "validation_data_path": if SKIP_EARLY_STOPPING then null else DEV_DATA_PATH,
  "test_data_path": if EVALUATE_ON_TEST then TEST_DATA_PATH else if SKIP_EARLY_STOPPING then DEV_DATA_PATH else null,
  "evaluate_on_test" : if EVALUATE_ON_TEST then EVALUATE_ON_TEST else if SKIP_EARLY_STOPPING then SKIP_EARLY_STOPPING else false,
  "model": {
    "type": "bert_crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": DROPOUT,
    "include_start_end_transitions": false,
    "text_field_embedder": PRETRAINED_ROBERTA_FIELDS(true)['embedder'],
  },
  "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": BATCH_SIZE
    },
    "validation_iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 64
 },
  "trainer": {
    "optimizer": PRETRAINED_ROBERTA_FIELDS(true)['optimizer'],
    "validation_metric": "+f1-measure-overall",
    "num_epochs": NUM_EPOCHS,
    "patience": PATIENCE,
    "gradient_accumulation_batch_size": GRAD_ACC,
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
  } + if SKIP_EARLY_STOPPING then {"checkpointer": PRETRAINED_ROBERTA_FIELDS(true)['checkpointer']} else {"num_serialized_models_to_keep": 0}
}

