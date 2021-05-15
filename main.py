import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Model:
    def __init__(self, dimx=None, dimh1=None, dimh2=None, dimy=None, step_size=None):
        self.stepSize = step_size

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

    def forward(self, x):
        # get the number of inputs
        self.m = x.shape[1]

        # first hidden layer
        self.h1 = np.maximum(0, np.matmul(self.W1, x))

        # second hidden layer
        self.h2 = np.maximum(0, np.matmul(self.W2, self.h1))

        # y calculation
        y = np.matmul(self.W3, self.h2)

        # softmax calculation
        tmp = np.exp(y)
        prediction = np.minimum(np.divide(tmp, np.sum(tmp, axis=0)), 0.99)

        return prediction

    def backward(self, x, prob, labels):
        # back propagation
        # from loss function to probabilities
        true_label_prob = prob[labels, np.arange(prob.shape[1])]
        grad_log = np.divide(1, np.multiply(-np.log(10), true_label_prob))

        # from probabilities to outputs
        grad_y = np.multiply(-true_label_prob, prob)
        grad_y[labels, np.arange(prob.shape[1])] += true_label_prob
        grad_y = np.multiply(grad_y, grad_log)

        # from outputs to W3
        self.gradW3 = np.moveaxis(np.broadcast_to(grad_y, (self.d2, self.K, self.m)), 0, 1)
        self.gradW3 = np.mean(np.multiply(self.gradW3, np.broadcast_to(self.h2, (self.K, self.d2, self.m))), axis=-1)

        # from outputs to h2
        grad_h2 = np.matmul(np.transpose(self.W3), grad_y)
        grad_h2 = np.multiply(np.where(self.h2 > 0, 1, 0), grad_h2)

        # from h2 to W2
        self.gradW2 = np.moveaxis(np.broadcast_to(grad_h2, (self.d1, self.d2, self.m)), 0, 1)
        self.gradW2 = np.mean(np.multiply(self.gradW2, np.broadcast_to(self.h1, (self.d2, self.d1, self.m))), axis=-1)

        # from h2 to h1
        grad_h1 = np.matmul(np.transpose(self.W2), grad_h2)
        grad_h1 = np.multiply(np.where(self.h1 > 0, 1, 0), grad_h1)

        # from h1 to W1
        self.gradW1 = np.moveaxis(np.broadcast_to(grad_h1, (self.n, self.d1, self.m)), 0, 1)
        self.gradW1 = np.mean(np.multiply(self.gradW1, np.broadcast_to(x, (self.d1, self.n, self.m))), axis=-1)

    def step(self):
        self.W1 -= self.stepSize*self.gradW1
        self.W2 -= self.stepSize*self.gradW2
        self.W3 -= self.stepSize*self.gradW3


def loss_function(prob, labels):
    # calculating loss function
    error = -np.mean(np.log10(prob[labels, np.arange(prob.shape[1])]))

    return error


def accuracy_calc(results, truth):

    prediction = results.argmax(axis=0)
    acc = np.sum((prediction == truth))/len(truth)

    return int(acc*100)


if __name__ == '__main__':
    
    pathToTrainingData = "digit-recognizer/train.csv"

    # getting the training data
    df = pd.read_csv(pathToTrainingData)

    label = df["label"].to_numpy()  # get labels

    df.drop(["label"], axis="columns", inplace=True)  # drop the labels, get the image data
    dataset = df.to_numpy()

    # splitting the data into training and validation sets
    trainSet, valSet, labelTrainSet, labelValSet = train_test_split(dataset, label, train_size=0.8, random_state=42, stratify=label)
    trainSet = np.transpose(trainSet)
    trainSet = np.true_divide(trainSet, 255)  # normalization

    valSet = np.transpose(valSet)
    valSet = np.true_divide(valSet, 255)  # normalization

    # get the batch size
    batchSize = 32

    # initializations for model
    model = Model(dimx=trainSet.shape[0], dimh1=50, dimh2=20, dimy=10, step_size=0.1)

    # initialization for measurements
    accuracyTrain = []
    accuracyVal = []
    lossTrain = []
    lossVal = []

    loss = 0
    accuracy = 0

    # iterations
    trainBatchIter = trainSet.shape[1]//batchSize + (trainSet.shape[1] % batchSize > 0)
    valBatchIter = valSet.shape[1]//batchSize + (valSet.shape[1] % batchSize > 0)
    numIter = 3

    for k in range(numIter):
        print(k)
        accuracy = 0
        loss = 0

        for i in range(trainBatchIter):
            LB = i*batchSize  # Lower Bound
            UB = min((i+1)*batchSize, trainSet.shape[1])  # Upper Bound
            Y_train = model.forward(trainSet[:, LB:UB])  # forward calculation
            loss += loss_function(Y_train, labelTrainSet[LB:UB])  # loss calculation
            model.backward(trainSet[:, LB:UB], Y_train,  labelTrainSet[LB:UB])  # back propagation
            model.step()  # take the step to the optimum

            accuracy += accuracy_calc(Y_train, labelTrainSet[LB:UB])  # Accuracy calculation on the training

        accuracyTrain.append(accuracy/trainBatchIter)
        lossTrain.append(loss/trainBatchIter)

        accuracy = 0
        loss = 0

        for i in range(valBatchIter):
            LB = i*batchSize  # Lower Bound
            UB = min((i+1)*batchSize, valSet.shape[1])  # Upper Bound
            Y_val = model.forward(valSet[:, LB:UB])  # forward calculation
            loss += loss_function(Y_val, labelValSet[LB:UB])  # loss calculation

            accuracy += accuracy_calc(Y_val, labelValSet[LB:UB])  # Accuracy calculation on the validation

        accuracyVal.append(accuracy/valBatchIter)
        lossVal.append(loss/valBatchIter)

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
