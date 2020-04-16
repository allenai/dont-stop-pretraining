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
local GRAD_ACC = std.parseInt(std.extVar("NUM_GRAD_ACC_STEPS"));
local LR_SCHEDULE = std.parseInt(std.extVar("LR_SCHEDULE")) == 1;

// skip early stopping? turning this on will prevent dev eval at each epoch.
local SKIP_EARLY_STOPPING = std.parseInt(std.extVar("SKIP_EARLY_STOPPING")) == 1;
local SKIP_TRAINING = std.parseInt(std.extVar("SKIP_TRAINING")) == 1;
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

local TRAIN_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "train." + JACKKNIFE_EXT else DATA_DIR  + "train.jsonl";
local DEV_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "dev." + JACKKNIFE_EXT else DATA_DIR  + "dev.jsonl";
local TEST_DATA_PATH = if JACKKNIFE then DATA_DIR + "jackknife/" + "dev." + JACKKNIFE_EXT else DATA_DIR  + "test.jsonl";

// ----------------------------
// INPUT EMBEDDING
// ----------------------------

local PRETRAINED_ROBERTA_FIELDS(TRAINABLE) = {
  "indexer": {
    "tokens": {
        "type": "pretrained_transformer",
        "model_name": "roberta-base",
        "max_length": 512
    }
  },
  "embedder": {
    "token_embedders": {
      "tokens":{
        "type": "pretrained_transformer",
        "model_name": MODEL_NAME,
        "max_length": 512
      }
    }
  },
  "tokenizer": {
    "type": "pretrained_transformer",
    "model_name": "roberta-base",
    "max_length": 512
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
  "checkpointer": {
    "type": "roberta_default",
    "num_epochs": NUM_EPOCHS,
    "skip_early_stopping": SKIP_EARLY_STOPPING
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
        "type": "text_classification_json_with_sampling",
        "lazy": LAZY,
        "tokenizer": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['tokenizer'],
        "max_sequence_length": 512,
        "token_indexers": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['indexer'],
    } + if TRAIN_THROTTLE > -1 then {"sample": TRAIN_THROTTLE} else {},
    "validation_dataset_reader": {
        "type": "text_classification_json_with_sampling",
        "lazy": LAZY,
        "tokenizer": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['tokenizer'],
        "max_sequence_length": 512,
        "token_indexers": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['indexer'],
    },
    "train_data_path": TRAIN_DATA_PATH,
    "validation_data_path": if SKIP_EARLY_STOPPING then null else DEV_DATA_PATH,
    "test_data_path": if EVALUATE_ON_TEST then TEST_DATA_PATH else if SKIP_EARLY_STOPPING then DEV_DATA_PATH else null,
    "evaluate_on_test" : if EVALUATE_ON_TEST then EVALUATE_ON_TEST else if SKIP_EARLY_STOPPING then SKIP_EARLY_STOPPING else false,
    "model": {
        "type": "basic_classifier_with_f1",
        "text_field_embedder": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['embedder'],
        "seq2vec_encoder": CLS_FIELDS(ROBERTA_EMBEDDING_DIM),
        "feedforward_layer": {
            "input_dim": ENCODER_OUTPUT_DIM,
            "hidden_dims": ROBERTA_EMBEDDING_DIM * FEEDFORWARD_WIDTH_MULTIPLIER,
            "num_layers": NUM_FEEDFORWARD_LAYERS,
            "activations": "tanh"
        },
        "dropout": DROPOUT
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": BATCH_SIZE
        }
    },
    "trainer": {
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
        "validation_metric": "+f1",
        "checkpointer": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['checkpointer'],
        "optimizer": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['optimizer'],
        "num_gradient_accumulation_steps": GRAD_ACC
    } + if SKIP_TRAINING then {"type": "no_op"} else {}
      + if LR_SCHEDULE then {"learning_rate_scheduler": PRETRAINED_ROBERTA_FIELDS(ROBERTA_TRAINABLE)['scheduler']} else {}
}


