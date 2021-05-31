import torch
import numpy as np
from CustomDataset import CustomDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torch import nn
from NN2Layer import NN2Layer
from matplotlib import pyplot as plt


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    train_loss /= size

    print("Training loss: {:.4f}  accuracy: {:.0f}%  \n".format(train_loss, 100*correct))

    return train_loss, (100*correct)


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print("Training loss: {:.4f}  accuracy: {:.0f}%  \n".format(test_loss, 100*correct))

    return test_loss, (100*correct)


if __name__ == "__main__":

    # check for the cuda device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # getting the dataset
    trainDataPath = "../digit-recognizer/train.csv"
    allTrainDataset = CustomDataset(
        trainDataPath,
        transform=ToTensor()
    )
    trainDatasetSize = int(len(allTrainDataset)*0.8)
    trainDataset, valDataset = random_split(allTrainDataset, [trainDatasetSize, len(allTrainDataset)-trainDatasetSize],
                                            generator=torch.Generator().manual_seed(42))

    # getting the model
    model = NN2Layer().to(device)

    # hyper parameters
    learingRate = 1e-3
    batchSize = 64
    epochs = 20

    # use DataLoader for efficiency
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=5)
    valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=5)

    # initializing loss function
    loss = nn.CrossEntropyLoss()

    # initializing optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learingRate, weight_decay=1e-5)

    trainAcc = []
    trainLoss = []

    valAcc = []
    valLoss = []

    tmpAcc = 0
    tmpLoss = 0

    for t in range(epochs):
        print(f"epoch {t+1} \n-----------------------")
        tmpLoss, tmpAcc = train_loop(trainDataLoader, model, loss, optimizer, device)
        trainLoss.append(tmpLoss)
        trainAcc.append(tmpAcc)

        tmpLoss, tmpAcc = test_loop(valDataLoader, model, loss, device)
        valLoss.append(tmpLoss)
        valAcc.append(tmpAcc)
    print("Done!!")

    torch.save(model, "model.pth")

    plt.plot(np.arange(epochs), trainAcc)
    plt.plot(np.arange(epochs), valAcc)
    plt.title("Accuracy vs Iteration")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["Training set", "Validation set"])
    plt.show()

    plt.plot(np.arange(epochs), trainLoss)
    plt.plot(np.arange(epochs), valLoss)
    plt.title("Error value vs Iteration")
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.legend(["Training set", "Validation set"])
    plt.show()