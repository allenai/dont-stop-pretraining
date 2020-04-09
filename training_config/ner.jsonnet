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
// gradient accumulation batch size
local GRAD_ACC = std.parseInt(std.extVar("NUM_GRAD_ACC_STEPS"));
// skip early stopping? turning this on will prevent dev eval at each epoch.
local SKIP_EARLY_STOPPING = std.parseInt(std.extVar("SKIP_EARLY_STOPPING")) == 1;
local SKIP_TRAINING = std.parseInt(std.extVar("SKIP_TRAINING")) == 1;
// are we jackknifing? only for hyperpartisan.
local JACKKNIFE = std.parseInt(std.extVar("JACKKNIFE")) == 1;
// jacknife file extension. only for hyperpartisan.
local JACKKNIFE_EXT = std.extVar("JACKKNIFE_EXT");
// embedding to use
local EMBEDDING = std.extVar("EMBEDDING");
// freeze embedding?
local FREEZE_EMBEDDING = std.parseInt(std.extVar("FREEZE_EMBEDDING")) == 0;
// width multiplier on hidden size of feedforward network
local FEEDFORWARD_WIDTH_MULTIPLIER = std.parseInt(std.extVar("FEEDFORWARD_WIDTH_MULTIPLIER"));
// number of feedforward layers
local NUM_FEEDFORWARD_LAYERS = std.parseInt(std.extVar("NUM_FEEDFORWARD_LAYERS"));
// early stopping patience
local PATIENCE = std.parseInt(std.extVar("PATIENCE"));


local TRAIN_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "train." + JACKKNIFE_EXT else DATA_DIR  + "train.txt";
local DEV_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "dev." + JACKKNIFE_EXT else DATA_DIR  + "dev.txt";
local TEST_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "dev." + JACKKNIFE_EXT else DATA_DIR  + "test.txt";


// fine-tuning checkpointer, prevents serialization model
local CHECKPOINTER = {
    "type": "fine-tuning",
    "num_epochs": NUM_EPOCHS,
};

local PRETRAINED_ROBERTA_FIELDS(TRAINABLE) = {
  "indexer": {
    "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": "roberta-base",
        "max_length": 512
    }
  },
  "embedder": {
    "token_embedders": {
      "tokens":{
        "type": "pretrained_transformer_mismatched",
        "model_name": MODEL_NAME,
        "max_length": 512
      }
    }
  },
  "optimizer": {
        "type": "huggingface_adamw_str_lr",
        "lr": LEARNING_RATE,
        "betas": [0.9, 0.98],
        "eps": 1e-6,
         "weight_decay": 0.1,
         "parameter_groups": [
         [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}, []],
     ]
  },
  "scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
  "embedding_dim": 768
};


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
  "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": BATCH_SIZE
        }
    },
  "trainer": {
    "optimizer": PRETRAINED_ROBERTA_FIELDS(true)['optimizer'],
    "validation_metric": "+f1-measure-overall",
    "num_epochs": NUM_EPOCHS,
    "patience": PATIENCE,
    "num_gradient_accumulation_steps": GRAD_ACC,
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
    "learning_rate_scheduler": PRETRAINED_ROBERTA_FIELDS(true)['scheduler']
  } + if SKIP_EARLY_STOPPING then {"checkpointer": CHECKPOINTER} else {}
      + if SKIP_TRAINING then {"type": "no_op"} else {}
}