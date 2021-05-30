import torch
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

        # loss and accuracy calculation
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

    correct /= size
    train_loss /= size

    print("Training loss:" + str(train_loss) + "   accuracy:" + str(correct*100) + "%  \n")


def test_loop(dataloader, model, loss_fn):
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
    print("Test loss:" + str(test_loss) + "   accuracy:" + str(correct*100) + "%  \n")


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
    batchSize = 8
    epochs = 10

    # use DataLoader for efficiency
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True)

    # initializing loss function
    loss = nn.CrossEntropyLoss()

    # initializing optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learingRate)

    for t in range(epochs):
        print(f"epoch {t+1} \n-----------------------")
        train_loop(trainDataLoader, model, loss, optimizer, device)
        test_loop(valDataLoader, model, loss)
    print("Done!!")