
MODEL_NAME = "google/flan-t5-base"

TRAIN_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/train-00000-of-00001.parquet"
VALIDATION_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/validation-00000-of-00001.parquet"
TEST_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/test-00000-of-00001.parquet"

INPUT_COLUMN = "radiology_report"
TARGET_COLUMN = "layman_report"

MODEL_PROMPT = "translate this radiology report into a summary for a layperson: "

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.3

EPOCHS = 3
LEARNING_RATE = 3e-4
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 128
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q", "v"]

OUTPUT_DIR = "t5-base-lora-tuned"
LOSS_OUT = "loss.png"