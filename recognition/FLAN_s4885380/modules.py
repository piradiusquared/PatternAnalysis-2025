from typing import Tuple

from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType

from dataset import *
from constants import *

class FlanModel:
    def __init__(self):
        pass

    def build(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        # Load actual Flan-T5 models
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # Load LoRA
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer

    def setup_optimiser(self, model, train_dataloader) -> Tuple[AdamW, get_scheduler]:
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        num_training_steps = EPOCHS * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        return optimizer, lr_scheduler

