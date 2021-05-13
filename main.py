import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Model:
    def __init__(self):
        self.loss = 0
        self.stepSize = 0.1

        self.m = 10
        self.n = 784
        self.d1 = 50
        self.d2 = 20
        self.K = 10

        self.W1 = np.random.randn(self.d1, self.n)*np.sqrt(2/self.n)
        self.W2 = np.random.randn(self.d2, self.d1)*np.sqrt(2/self.d1)
        self.W3 = np.random.randn(self.K, self.d2)*np.sqrt(2/self.d2)

        self.h2 = np.zeros(self.d1)
        self.h1 = np.zeros(self.d2)

        self.gradW1 = np.zeros((self.d1, self.n))
        self.gradW2 = np.zeros((self.d2, self.d1))
        self.gradW3 = np.zeros((self.K, self.d2))

        self.prob = np.zeros((self.K, self.m))

    def forward(self, X):
        # get the # of inputs
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

    return acc


if __name__ == '__main__':
    
    pathToTrainingData = "digit-recognizer/train.csv"

    # getting the training data
    df = pd.read_csv(pathToTrainingData)

    label = df["label"].to_numpy()  # get labels

    df.drop(["label"], axis="columns", inplace=True)  # drop the labels, get the image data
    dataset = df.to_numpy()

    # splitting the data into training and validation sets
    trainSet, validationSet, labelTrainSet, labelValSet = train_test_split(dataset, label, train_size=0.8, random_state=42, stratify=label)

    # Training
    sampleNum = 5
    trainSet = np.true_divide(trainSet, 255)  # normalization
    X = np.transpose(trainSet[0:sampleNum])
    Y = labelTrainSet[0:sampleNum]

    # initializations for model
    model = Model()
    numIter = 1000
    accuracyTrain = np.zeros(numIter)
    accuracyTest = np.zeros(numIter)
    lossResults = np.zeros(numIter)

    # iterations
    for i in range(numIter):

        Y_predict = model.forward(X)  # forward calculation

        model.loss_function(Y_predict, Y)  # loss calculation

        model.backward(X, Y)  # back propagation

        model.step()  # take the step to the optimum

        accuracyTrain[i] = accuracy(Y_predict, Y)  # Accuracy calculation on the training and validation

        lossResults[i] = model.loss

        print(i, "  ", accuracyTrain[i], "  ", model.loss)

    plt.plot(np.arange(numIter), accuracyTrain)
    plt.title("Accuracy on training data vs Iteration (with " + str(sampleNum) + " samples and " + str(numIter) + " iterations)")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.show()

    plt.plot(np.arange(numIter), lossResults)
    plt.title("Error value vs Iteration (with " + str(sampleNum) + " samples and " + str(numIter) + " iterations)")
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()

    # pd.DataFrame(model.W1).to_csv("W1.csv")
    # pd.DataFrame(model.W2).to_csv("W2.csv")
    # pd.DataFrame(model.W3).to_csv("W3.csv")
