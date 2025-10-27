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


    def train(self):
        pass

    def train_epoch(self, epoch: int) -> None:
        pass


    def evaluate_epoch(self, epoch):
        pass