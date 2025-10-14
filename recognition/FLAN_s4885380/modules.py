import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


print(dataset["train"].column_names)

def preprocess_function(batch):
    prefix = "translate this radiology report into a summary for a layperson: "
    inputs = [prefix + report for report in batch["radiology_report"]]
    targets = [report for report in batch["layman_report"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

