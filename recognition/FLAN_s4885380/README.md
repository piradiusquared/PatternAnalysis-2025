# Fine-Tuning a Flan-T5 Model for Layman Summarisation

> Author: Yufan Pan  
> ID: 4885380

## Flan-T5

### Problem

### Architecture

## Dataset Loading

## Training Specification

## Output

### Rouge Score

## Reproducing Requirements
### Implementation Details
| Parameter | Description |
| --------- | ----------- |
| Model     | `T5-Base`   |
| Parameter count | Base Model: ~250M, Trainable: (Find and UPDATE) |
| Fine-Tuning Strategy | LoRA (INSERT specifications) |
| Epochs    |             |
| Learning Rate |          |
| Training Time | 10 BILLION hours (UPDATE later) | 

### Hardware Specification:
| Hardware | Description |
| -------- | ----------- |
| CPU      | 8x vCPU cores (AMD Zen 2) |
| GPU Type | NVDIA A100 |
| VRAM     | 40 GB |
| RAM      | 64 GB |


### Library Installation:
The following libraries are required to reproduce the fine-tuning process:
``` Bash
torch
transformers
datasets
evaluate
peft
accelerate
bitsandbytes
sentencepiece
rouge-score
```
Or, you may install through the provided ```requirements.txt``` file by running:
``` Bash
pip install -r requirements.txt
```