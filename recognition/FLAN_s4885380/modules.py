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

from datasets import Dataset, DatasetDict

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
dataset_test = DatasetDict(test_data)
tokenised_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['radiology_report', 'layman_report'])

print("Dataset preprocessed successfully!")
print(tokenised_dataset['train'][0].keys())