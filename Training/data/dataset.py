import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, vectorizer):
        self.df = pd.read_csv(csv_file)
        self.vectorizer = vectorizer
        self._lookup = {split: self.df[self.df.split == split] for split in ["train", "val", "test"]}
        self.set_split("train")

    def set_split(self, split):
        self._target_df = self._lookup[split]
        self._size = len(self._target_df)

    def __getitem__(self, idx):
        row = self._target_df.iloc[idx]
        text = str(row["comment_text"])
        vec = self.vectorizer.vectorize(text, max_length=100)
        target = row["target"]
        return torch.tensor(vec), torch.tensor(target)

    def __len__(self):
        return self._size