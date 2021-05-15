import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Model:
    def __init__(self, dimx, dimh1, dimh2, dimy):
        self.loss = 0
        self.stepSize = 0.1

        self.m = 0
        self.n = dimx
        self.d1 = dimh1
        self.d2 = dimh2
        self.K = dimy

        self.W1 = np.random.randn(self.d1, self.n)*np.sqrt(2/self.n)
        self.W2 = np.random.randn(self.d2, self.d1)*np.sqrt(2/self.d1)
        self.W3 = np.random.randn(self.K, self.d2)*np.sqrt(2/self.d2)

        self.h2 = np.zeros(self.d1)
        self.h1 = np.zeros(self.d2)

        self.gradW1 = np.zeros((self.d1, self.n))
        self.gradW2 = np.zeros((self.d2, self.d1))
        self.gradW3 = np.zeros((self.K, self.d2))

        self.prob = np.array(None)

    def forward(self, X):
        # get the number of inputs
        self.m = X.shape[1]

        # first hidden layer
        self.h1 = np.maximum(0, np.matmul(self.W1, X))

        # second hidden layer
        self.h2 = np.maximum(0, np.matmul(self.W2, self.h1))

        # the result
        prediction = np.matmul(self.W3, self.h2)

        return prediction

    def backward(self, X, labels):
        # back propagation
        # from loss function to probabilities
        prob_true = self.prob[labels, np.arange(self.prob.shape[1])]
        grad_log = np.divide(1, np.multiply(-np.log(10), prob_true))

        # from probabilities to outputs
        grad_Y = np.multiply(-prob_true, self.prob)
        grad_Y[labels, np.arange(self.prob.shape[1])] += prob_true
        grad_Y = np.multiply(grad_Y, grad_log)

        # from outputs to W3
        self.gradW3 = np.moveaxis(np.broadcast_to(grad_Y, (self.d2, self.K, self.m)), 0, 1)
        self.gradW3 = np.mean(np.multiply(self.gradW3, np.broadcast_to(self.h2, (self.K, self.d2, self.m))), axis=-1)

        # from outputs to h2
        grad_h2 = np.matmul(np.transpose(self.W3), grad_Y)
        grad_h2 = np.multiply(np.where(self.h2 > 0, 1, 0), grad_h2)

        # from h2 to W2
        self.gradW2 = np.moveaxis(np.broadcast_to(grad_h2, (self.d1, self.d2, self.m)), 0, 1)
        self.gradW2 = np.mean(np.multiply(self.gradW2, np.broadcast_to(self.h1, (self.d2, self.d1, self.m))), axis=-1)

        # from h2 to h1
        grad_h1 = np.matmul(np.transpose(self.W2), grad_h2)
        grad_h1 = np.multiply(np.where(self.h1 > 0, 1, 0), grad_h1)

        # from h1 to W1
        self.gradW1 = np.moveaxis(np.broadcast_to(grad_h1, (self.n, self.d1, self.m)), 0, 1)
        self.gradW1 = np.mean(np.multiply(self.gradW1, np.broadcast_to(X, (self.d1, self.n, self.m))), axis=-1)

    def loss_function(self, Y, labels):
        # calculating probabilities for all y
        tmp = np.exp(Y)
        self.prob = np.minimum(np.divide(tmp, np.sum(tmp, axis=0)), 0.99)

        # calculating loss function
        self.loss = -np.mean(np.log10(self.prob[labels, np.arange(self.prob.shape[1])]))

    def step(self):
        self.W1 -= self.stepSize*self.gradW1
        self.W2 -= self.stepSize*self.gradW2
        self.W3 -= self.stepSize*self.gradW3


def accuracy(calculatedY, label):

    prediction = calculatedY.argmax(axis=0)

    acc = np.sum((prediction == label))/len(label)

    return int(acc*100)


if __name__ == '__main__':
    
    pathToTrainingData = "digit-recognizer/train.csv"

    # getting the training data
    df = pd.read_csv(pathToTrainingData)

    label = df["label"].to_numpy()  # get labels

    df.drop(["label"], axis="columns", inplace=True)  # drop the labels, get the image data
    dataset = df.to_numpy()

    # splitting the data into training and validation sets
    trainSet, validationSet, labelTrainSet, labelValSet = train_test_split(dataset, label, train_size=0.8, random_state=42, stratify=label)
    trainSet = np.transpose(trainSet)
    trainSet = np.true_divide(trainSet, 255)  # normalization

    validationSet = np.transpose(validationSet)
    validationSet = np.true_divide(validationSet, 255)  # normalization

    # split the training set (it is too big)
    splitBy = 100
    trainingBatchList = np.split(trainSet, splitBy, axis=1)
    trainingBatchLabel = np.split(labelTrainSet, splitBy)
    valBatchList = np.split(validationSet, splitBy, axis=1)
    valBatchLabel = np.split(labelValSet, splitBy)

    # initializations for model
    model = Model(784, 50, 20, 10)

    # initialization for measurements
    numIter = 20

    accuracyTrain = []
    accuracyVal = []
    lossTrain = []
    lossVal = []

    loss = 0
    acc = 0

    # iterations
    for k in range(numIter):
        print(k)
        acc = 0
        loss = 0

        for i in range(len(trainingBatchList)):
            Y_train = model.forward(trainingBatchList[i])  # forward calculation
            model.loss_function(Y_train, trainingBatchLabel[i])  # loss calculation
            model.backward(trainingBatchList[i], trainingBatchLabel[i])  # back propagation
            model.step()  # take the step to the optimum

            acc += accuracy(Y_train, trainingBatchLabel[i])  # Accuracy calculation on the training
            loss += model.loss # Loss calculation on the training

        accuracyTrain.append(acc/len(trainingBatchList))
        lossTrain.append(loss/len(trainingBatchList))

        acc = 0
        loss = 0

        for i in range(len(valBatchList)):
            Y_val = model.forward(valBatchList[i])  # forward calculation
            model.loss_function(Y_val, valBatchLabel[i])  # loss calculation

            acc += accuracy(Y_val, valBatchLabel[i])  # Accuracy calculation on the validation
            loss += model.loss  # loss calculation on the validation

        accuracyVal.append(acc/len(valBatchList))
        lossVal.append(loss/len(valBatchList))

    plt.plot(np.arange(numIter), accuracyTrain)
    plt.plot(np.arange(numIter), accuracyVal)
    plt.title("Accuracy vs Iteration")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.legend(["Training set", "Validation set"])
    plt.show()

    plt.plot(np.arange(numIter), lossTrain)
    plt.plot(np.arange(numIter), lossVal)
    plt.title("Error value vs Iteration")
    plt.ylabel("Error")
    plt.xlabel("Iteration")
    plt.legend(["Training set", "Validation set"])
    plt.show()

    np.save("W3.csv", model.W3)
    np.save("W2.csv", model.W2)
    np.save("W1.csv", model.W1)
