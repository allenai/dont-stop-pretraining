{
    "SEED": 42,
    "DATASET": "citation_intent",
    "MODEL_NAME": "roberta-base",
    "DATA_DIR": "s3://suching-dev/textcat/science/citation_intent/",
    "EVALUATE_ON_TEST": true,
    "TRAIN_THROTTLE": -1,
    "LAZY": false,
    "JACKKNIFE": false,
    "SKIP_EARLY_STOPPING": false,
    "SKIP_TRAINING": false,
    "LEARNING_RATE": {
        "sampling strategy": "loguniform",
        "bounds": [1e-6, 1e-4]
    },
    "DROPOUT": {
        "sampling strategy": "uniform",
        "bounds": [0, 1]
    },
    "ENCODER": "CLS",
    "NUM_FEEDFORWARD_LAYERS": 1,
    "FEEDFORWARD_WIDTH_MULTIPLIER": 1,
    "EMBEDDING": "ROBERTA",
    "NUM_EPOCHS": {
        "sampling strategy": "integer",
        "bounds": [1, 10]
    },
    "PATIENCE": 3,
    "GRAD_ACC_BATCH_SIZE": 16,
    "BATCH_SIZE": 16,
    "CUDA_DEVICE": 0
}


