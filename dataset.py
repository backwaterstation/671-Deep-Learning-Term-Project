# dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset

class GHCNDailyDataset(Dataset):
    def __init__(self, dataframe, feature="TMAX", sequence_length=7):
        self.sequence_length = sequence_length
        self.feature = feature

        # filter only the feature of interest
        self.data = dataframe[dataframe["datatype"] == self.feature].copy()

        # sort by date
        self.data.sort_values("date", inplace=True)

        # extract just the value column
        self.values = self.data["value"].values

        # optional: normalize values
        self.values = (self.values - self.values.mean()) / self.values.std()

        self.num_samples = len(self.values) - self.sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        if idx + self.sequence_length >= len(self.values):
            raise IndexError("Index out of range for sequence window.")

        # select the 'value' column only for x (sequence)
        x = self.values[idx:idx + self.sequence_length]
        y = self.values[idx + self.sequence_length]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
