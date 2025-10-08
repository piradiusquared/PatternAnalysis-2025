import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Testing for dataset loading
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


print(dataset["train"].column_names)
