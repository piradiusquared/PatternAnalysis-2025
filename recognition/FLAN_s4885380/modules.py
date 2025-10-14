import torch
from datasets import load_dataset, Dataset
from peft import LoraModel, LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Constant values
FLAN_MODEL = "google/flan-t5-base"
DATASET_URL = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"

MAX_INPUT = 1024 # Sufficient length
MAX_LABEL = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)

def preprocess_function(batch):
    prefix = "translate this radiology report into a summary for a layperson: "
    inputs = [prefix + report for report in batch["radiology_report"]]
    targets = [report for report in batch["layman_report"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

test_data = {
    'train': Dataset.from_dict({
        'radiology_report': ["The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax."],
        'layman_report': ["The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs."]
    }),
    'validation': Dataset.from_dict({
        'radiology_report': ["Central venous catheter traversing the left jugular vein with its tip in the superior vena cava. The remainder is unchanged."],
        'layman_report': ["A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before."]
    }),
    'test': Dataset.from_dict({
        'radiology_report': ["Chronic pulmonary changes"],
        'layman_report': ["Long-term changes in the lungs are seen."]
    })
}
biolay_dataset = load_dataset(DATASET_URL)
original_columns = biolay_dataset["train"].column_names
tokenised_dataset = biolay_dataset.map(preprocess_function, batched=True, remove_columns=original_columns)

print("Dataset preprocessed successfully!")
print(tokenised_dataset["train"][0].keys())

config = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
)

model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL)
lora_model = LoraModel(model, config, "default")