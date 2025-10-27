import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from constants import *

class SplitData:
    def __init__(self, file_path: str) -> None:
        self.dataframe = pd.read_parquet(file_path)
        self.dataframe = self.dataframe[0:200] # Remove when actually training

    def get_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_index = np.random.random(len(self.dataframe)) < 0.7
        train = self.dataframe[split_index]
        validation = self.dataframe[~split_index]

        return train, validation
        

class FlanDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.prefix = "translate this radiology report into a summary for a layperson: "

        self.dataframe = dataframe

        # Biolaysumm dataset is of .parquet file type
        # Future addition: add support for basic files
        # self.dataframe = pd.read_parquet(file_path)
        # self.dataframe = self.dataframe[0:50] # Slice data for subset

    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, index: int) -> list:
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
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# dataframe = SplitData(file_path=TRAIN_FILE)
# train, validation = dataframe.get_splits()

# print(len(train))
# print(len(validation))
