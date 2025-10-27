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
        self.scaler = GradScaler() # New version doesn't work on rangpur
        self.metric = evaluate.load("rouge")

        self._train_loss = [] # Loss list for plotting
        
    def train(self):
        for epoch in range(EPOCHS):
            self.train_epoch(epoch=epoch)

    """
    References: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
    https://huggingface.co/learn/llm-course/en/chapter3/4#next-steps-and-best-practices
    """
    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        train_progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1} Training")

        for batch in train_progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss

            self.scaler.scale(loss).backward() # Step optimiser and scalers
            self.scaler.step(optimizer=self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            self._train_loss.append(loss.item()) # Add loss per batch to list

            train_progress.set_postfix(loss=loss.item())
            train_progress.update(1)


    def evaluate_epoch(self, epoch):
        pass
    
    def get_train_loss(self) -> list:
        return self._train_loss