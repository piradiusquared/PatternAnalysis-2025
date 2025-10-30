import torch
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

BASE_MODEL = "google/flan-t5-base"
FINETUNED_MODEL = "t5-base-lora-tuned/epoch_3" # Take last epoch

base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

predict_dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
predict_dataset = predict_dataset.shuffle(seed=889)
random_predict = predict_dataset["validation"]

predictions = []
references = []

for i in range(5): # Number of comparisons
    radiology_report = random_predict[i]['radiology_report']
    layman_report = random_predict[i]['layman_report']

    prompt = f"translate this radiology report into a summary for a layperson: {radiology_report}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Get fine tuned model to generate 
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256) # constant for 256
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nExample {i + 1}")
    print(f"Official Layman Report: {layman_report}")
    print(f"Fine tuned Model's Layman Report: {prediction}")

    predictions.append(prediction)
    references.append(layman_report) # Official report from dataset

rouge_scores = evaluate.load("rouge")
scores = rouge_scores.compute(predictions=predictions, references=references, use_stemmer=True)

print(f"Final ROUGE scores after predictions")
for key, value in scores.items():
    print(f"{key}: {value * 100: .4f}")
