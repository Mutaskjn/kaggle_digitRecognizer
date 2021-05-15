import pandas as pd
import numpy as np
from NN2Layer import NN2Layer


if __name__ == '__main__':

    pathToTestData = "digit-recognizer/test.csv"

    df = pd.read_csv(pathToTestData)
    testSet = df.to_numpy()
    testSet = np.transpose(testSet)
    testSet = np.true_divide(testSet, 255)

    model = NN2Layer()

    model.W3 = np.load("weights/W3.npy")
    model.W2 = np.load("weights/W2.npy")
    model.W1 = np.load("weights/W1.npy")

    batchSize = 32
    testBatchIter = testSet.shape[1] // batchSize + (testSet.shape[1] % batchSize > 0)
    accuracy = 0

    predictions = np.zeros(testSet.shape[1], dtype=int)

    for i in range(testBatchIter):
        LB = i * batchSize  # Lower Bound
        UB = min((i + 1) * batchSize, testSet.shape[1])  # Upper Bound
        Y_test = model.forward(testSet[:, LB:UB])  # forward calculation
        predictions[LB:UB] = Y_test.argmax(axis=0).astype(int)

    result = pd.DataFrame(predictions)
    result.index.name = "ImageId"
    result.index += 1
    result.rename(columns={0: 'Label', -1: 'ImageId'}, inplace=True)

    result.to_csv("submission.csv")
