# File containing all constants used
MODEL_NAME = "google/flan-t5-base"

# Pandas dataframe link for BioLaySumm dataset
TRAIN_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/train-00000-of-00001.parquet"
VALIDATION_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/validation-00000-of-00001.parquet"
TEST_FILE = "hf://datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/data/test-00000-of-00001.parquet"

INPUT_COLUMN = "radiology_report"
TARGET_COLUMN = "layman_report"

# Prompt used exclusively in predict.py
MODEL_PROMPT = "translate this radiology report into a summary for a layperson: "

# Held-out splits
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.3

# Training parameters
EPOCHS = 3
LEARNING_RATE = 3e-4
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 128
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

# LoRA Parameters
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q", "v"]

# File paths for model saving and loss plotting
OUTPUT_DIR = "t5-base-lora-tuned"
LOSS_OUT = "loss.png"
