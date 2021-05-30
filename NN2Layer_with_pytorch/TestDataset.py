import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.length = pd.read_csv(data_path).shape[0]
        self.img_dir = data_path
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = np.array(pd.read_csv(self.img_dir).iloc[idx, :], dtype=np.float32).reshape(28, 28)
        if self.transform:
            image = self.transform(image)
        return image
