import torch
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

from constants import *

FINETUNED_MODEL = "t5-base-lora-tuned/epoch_3" # Take last epoch for best performance

"""
Computes the perplexity score using the model loss
"""
def perplexity_score(model: AutoModelForSeq2SeqLM,
                     tokenizer: AutoTokenizer,
                     prompt: str,
                     target_text: str,
                     device="cuda") -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    labels = tokenizer(target_text, return_tensors="pt").input_ids.to(device)

    # Gets the loss during benchmarking
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

    perplexity = torch.exp(loss) # Calculate perplexity
    return perplexity.item()

# Get new base flan-t5 model, and load in saved trained model
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
base_model.eval()

# Use completely fresh Flan-T5 model
new_t5 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
fine_tuned_model = PeftModel.from_pretrained(new_t5, FINETUNED_MODEL)
fine_tuned_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Use API for loading in dataset
predict_dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
predict_dataset = predict_dataset.shuffle(seed=3710)
random_predict = predict_dataset["validation"]

predictions = []
references = []

for i in range(5): # Number of evaluations
    radiology_report = random_predict[i]['radiology_report']
    layman_report = random_predict[i]['layman_report']

    prompt = f"translate this radiology report into a summary for a layperson: {radiology_report}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Get fine tuned model to generate a summary
    with torch.no_grad():
        outputs = fine_tuned_model.generate(**inputs, max_new_tokens=MAX_INPUT_LENGTH)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Compare perplexity
    fine_tune_perplexity = perplexity_score(fine_tuned_model, tokenizer, prompt, layman_report)
    base_model_perplexity = perplexity_score(base_model, tokenizer, prompt, layman_report)

    print(f"\nExample {i + 1}")
    print(f"Official Layman Report: {layman_report}")
    print(f"Fine tuned Model's Layman Report: {prediction}")

    print(f"\nFine Tuned Model Perplexity on Official Report: {fine_tune_perplexity:.4f}")
    print(f"\nBase Model Perplexity on Official Report: {base_model_perplexity:.4f}")

    predictions.append(prediction)
    references.append(layman_report) # Official report from dataset

rouge_scores = evaluate.load("rouge")
scores = rouge_scores.compute(predictions=predictions, references=references, use_stemmer=True)

print(f"Final ROUGE scores after predictions")
for key, value in scores.items():
    print(f"{key}: {value * 100: .4f}")
