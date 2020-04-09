{
    "SEED": 42,
    "DATASET": "ag",
    "MODEL_NAME": "roberta-base",
    "DATA_DIR": "s3://suching-dev/textcat/news/ag/",
    "EVALUATE_ON_TEST": true,
    "TRAIN_THROTTLE": -1,
    "LAZY": false,
    "JACKKNIFE": false,
    "SKIP_EARLY_STOPPING": false,
    "SKIP_TRAINING": false,
    "LEARNING_RATE": 2e-5,
    "DROPOUT": 0.1,
    "ENCODER": "CLS",
    "NUM_FEEDFORWARD_LAYERS": 1,
    "FEEDFORWARD_WIDTH_MULTIPLIER": 1,
    "EMBEDDING": "ROBERTA",
    "NUM_EPOCHS": 10,
    "PATIENCE": 3,
    "NUM_GRAD_ACC_STEPS": 4,
    "BATCH_SIZE": 16,
    "CUDA_DEVICE": 0
}


