import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class TestDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.imgs = np.array(pd.read_csv(data_path), dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.imgs[:, 0])

    def __getitem__(self, idx):
        image = self.imgs[idx, :].reshape(28, 28)
        if self.transform:
            image = self.transform(image)
        return image


def test_loop(dataloader, model):
    resultList = []

    with torch.no_grad():
        for X in dataloader:

            pred = model(X)
            resultList.append(pred.argmax(1).numpy())

    result = np.concatenate(resultList, axis=None)

    return result


if __name__ == '__main__':

    pathToTestData = "../digit-recognizer/test.csv"

    testSet = TestDataset(
        pathToTestData,
        transform=ToTensor()
    )

    batchSize = 64
    testDataloader = DataLoader(testSet, batch_size=batchSize, num_workers=5)

    model = torch.load("model.pth").to('cpu')

    predictions = test_loop(testDataloader, model)

    result = pd.DataFrame(predictions)
    result.index.name = "ImageId"
    result.index += 1
    result.rename(columns={0: 'Label', -1: 'ImageId'}, inplace=True)

    result.to_csv("submission.csv")
