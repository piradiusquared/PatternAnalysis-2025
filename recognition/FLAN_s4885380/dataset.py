import random
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from constants import *

"""
Held-out data splitter for training and evaluation.
Splits the data into 70/30 ratio
"""
class SplitData:
    def __init__(self, file_path: str, sample_size: int | None = None) -> None:
        self.dataframe = pd.read_parquet(file_path)
        if sample_size != None:
            self.dataframe = self.dataframe[0:sample_size]
        # else:
        #     self.dataframe = self.dataframe[100:300] # Testing split

    """
    Returns both splits at once from the original dataframe
    """
    def get_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_index = np.random.random(len(self.dataframe)) < 0.7
        train = self.dataframe[split_index]
        validation = self.dataframe[~split_index]

        return train, validation
        

"""
Custom dataset loader and preprocessor. Prepends 1 of 4 similar prompts for training and evaluation.
"""
class FlanDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer) -> None:
        self.tokenizer = tokenizer
        # self.prefix = MODEL_PROMPT
        self._prompts = [
            "Translate this radiology report into a summary for a layperson: ",
            "Summarise the following medical report in simple, easy-to-understand terms: ",
            "Explain this radiology report to a patient with no medical background: ",
            "Provide a layperson's summary for this report: "
        ]

        self.dataframe = dataframe

        # Biolaysumm dataset is of .parquet file type
        # Future addition: add support for basic files
        # self.dataframe = pd.read_parquet(file_path)
        # self.dataframe = self.dataframe[0:50] # Slice data for subset

    def __len__(self) -> int:
        return len(self.dataframe)

    """
    Tokenises the inputs using the tokenizer API. Converts strings into NLP suitable tensors
    """
    def __getitem__(self, index: int) -> list:
        row = self.dataframe.iloc[index] # Selects slices using iloc index

        rand_prefix = random.choice(self._prompts) # Selects random prefix
        report = rand_prefix + str(row[INPUT_COLUMN])
        summary = str(row[TARGET_COLUMN])

        model_inputs = self.tokenizer( # Tokenises the radiology report
            report,
            max_length=MAX_INPUT_LENGTH,
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer(): # Tokenises the layman summary
            labels = self.tokenizer(
                summary,
                max_length=MAX_TARGET_LENGTH,
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"] # Join report and summary together
        return model_inputs
