import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from constants import *


class FlanDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.prefix = "translate this radiology report into a summary for a layperson: "
    
        # Biolaysumm dataset is of .parquet file type
        self.dataframe = pd.read_parquet(file_path)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        report = self.prefix + str(row[INPUT_COLUMN])
        summary = str(row[TARGET_COLUMN])

        model_inputs = self.tokenizer(
            report,
            max_length=MAX_INPUT_LENGTH,
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summary,
                max_length=MAX_TARGET_LENGTH,
                truncation=True
            )

        return model_inputs
