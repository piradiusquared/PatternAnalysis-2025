
MODEL_NAME = "google/flan-t5-small"

TRAIN_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/train-00000-of-00001.parquet"
VALIDATION_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/validation-00000-of-00001.parquet"
TEST_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/test-00000-of-00001.parquet"

INPUT_COLUMN = "radiology_report"
TARGET_COLUMN = "layman_report"

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.3

EPOCHS = 3
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q", "v"]

OUTPUT_DIR = "t5-base-lora-tuned"