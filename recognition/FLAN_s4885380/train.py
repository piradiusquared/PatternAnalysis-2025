import time
import torch
import numpy as np
import evaluate
from transformers import DataCollatorForSeq2Seq
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import *
from modules import *
from constants import *

class FlanTrainer:
    def __init__(self,
                 model: AutoModelForSeq2SeqLM,
                 tokenizer: AutoTokenizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: AdamW,
                 lr_scheduler,
                 device: torch.device
                 ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.scaler = GradScaler()
        self.metric = evaluate.load("rouge")

        self._train_loss = []
        
    def train(self):
        start_time = time.time()
        for epoch in range(EPOCHS):
            epoch_start = time.time()

            # Train and eval per epoch
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} took {epoch_time/60 : .2f} minutes")

        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    def train_epoch(self, epoch: int) -> None:
        print(f"\nStarting Epoch {epoch+1}/{EPOCHS}")
        self.model.train()
        train_progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1} Training")

        batch_num = 0
        for batch in train_progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs.loss

            # Scale and optimise
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer=self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            # Add loss for plotting
            self._train_loss.append(loss.item())

            train_progress.set_postfix(loss=loss.item())
            tqdm.write(f"Batch: {batch_num} Loss: {loss.item(): .4f}")
            batch_num += 1


    def evaluate_epoch(self, epoch):
        print(f"Evaluation for Epoch {epoch + 1}")
        self.model.eval()

        all_preds = []
        all_labels = []

        eval_progress = tqdm(self.eval_dataloader, desc="Evaluating")
        for batch in eval_progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=MAX_TARGET_LENGTH,
                )
            
            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # -100 for padding reasons
            labels = np.where(batch["labels"].cpu() != -100, batch["labels"].cpu(), self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Add in decoded
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)
        
        # Compute rouge scores
        result = self.metric.compute(predictions=all_preds, references=all_labels, use_stemmer=True)
        result = {k: v * 100 for k, v in result.items()}
        print(f"Evaluation ROUGE scores for Epoch {epoch+1}:")
        print(f"rouge1: {result['rouge1']:.4f}, rouge2: {result['rouge2']:.4f}, rougeL: {result['rougeL']:.4f}, rougeLsum: {result['rougeLsum']:.4f}")
        
        epoch_output_dir = f"{OUTPUT_DIR}/epoch_{epoch + 1}"
        self.model.save_pretrained(epoch_output_dir)
        self.tokenizer.save_pretrained(epoch_output_dir)
        print(f"Epoch {epoch + 1} Model is saved to: {OUTPUT_DIR}/epoch_{epoch + 1}")
    
    def get_train_loss(self) -> list:
        return self._train_loss


"""
Actual training script
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

builder = FlanModel()

model, tokenizer = builder.build()
model.to(device)

# Preprocess data into splits:

dataframe = SplitData(file_path=TRAIN_FILE)
train_split, validation_split = dataframe.get_splits()

# Create Datasets and DataLoaders
train_dataset = FlanDataset(dataframe=train_split, tokenizer=tokenizer)
validation_dataset = FlanDataset(dataframe=validation_split, tokenizer=tokenizer)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Load in from Pandas
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=TRAIN_BATCH_SIZE
)

eval_dataloader = DataLoader(
    validation_dataset, shuffle=True, collate_fn=data_collator, batch_size=VALID_BATCH_SIZE
)

optimizer, scheduler = builder.setup_optimiser(model=model, train_dataloader=train_dataloader)
trainer = FlanTrainer(model=model,
                        tokenizer=tokenizer,
                        train_dataloader=train_dataloader,
                        eval_dataloader=eval_dataloader,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        device=device)

trainer.train()

import matplotlib.pyplot as plt
plt.plot(trainer.get_train_loss(), label="Train loss")
plt.xlabel("batch")
plt.ylabel("loss")
plt.legend()
plt.savefig(LOSS_OUT) # Save plot