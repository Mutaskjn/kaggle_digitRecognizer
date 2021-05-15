import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from NN2Layer import NN2Layer


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
    model = NN2Layer(input_dim=trainSet.shape[0], first_layer_dim=20, second_layer_dim=10,
                     output_dim=10, step_size=0.1, regularization_const=1e-5)

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
    numIter = 20

    for k in range(numIter):
        print(k, end=" ")
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
        print(loss/trainBatchIter, end=" ")
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
        print(loss/valBatchIter)

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

    # np.save("W3.csv", model.W3)
    # np.save("W2.csv", model.W2)
    # np.save("W1.csv", model.W1)
