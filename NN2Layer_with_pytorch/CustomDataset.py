import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(data_path).iloc[:, 0]
        self.imgs = np.array(pd.read_csv(data_path).iloc[:, 1:], dtype=np.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx, :].reshape(28, 28)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
