import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numba import jit, cuda

class Model:

    def __init__(self, inputs, labels):
        self.labels = labels

        self.X = inputs
        self.Y = []

        self.loss = 1
        self.stepSize = 1e-4

        self.W1 = np.random.rand(1000,784)*1e-4
        self.W2 = np.random.rand(500,1000)*1e-4
        self.W3 = np.random.rand(10,500)*1e-4

        self.B1 = np.random.rand(1000)*1e-4
        self.B2 = np.random.rand(500)*1e-4
        self.B3 = np.random.rand(10)*1e-4

        self.h2 = []
        self.h1 = []

        self.gradW1 = np.zeros((1000,784), float)
        self.gradW2 = np.zeros((500,1000), float)
        self.gradW3 = np.zeros((10,500), float)

        self.gradB1 = np.zeros(1000, float)
        self.gradB2 = np.zeros(500, float)
        self.gradB3 = np.zeros(10, float)

        self.prob = []
        self.count = 1

    def forward(self):
        #first hidden layer
        self.h1 = [np.matmul(self.W1, np.transpose(self.X[i,:]))+self.B1 for i in range(self.X.shape[0])]
        self.h1 = [np.maximum(0,self.h1[i]) for i in range(len(self.h1))]
        
        #second hidden layer
        self.h2 = [np.matmul(self.W2, self.h1[i])+self.B2 for i in range(len(self.h1))]
        self.h2 = [np.maximum(0,self.h2[i]) for i in range(len(self.h2))]
        
        #The result
        self.Y = [np.matmul(self.W3, self.h2[i])+self.B3 for i in range(len(self.h2))]

    def backward(self):
        #calculating gradient
        #from loss function to y
        logGrad = [np.multiply(-np.log(10), self.prob[i][self.labels[i]]) for i in range(len(self.prob))]
        self.Y = [np.where(self.prob[i]==self.prob[i][self.labels[i]], self.prob[i][self.labels[i]], 0)\
            -np.multiply(self.prob[i],self.prob[i][self.labels[i]]) for i in range(len(self.prob))]
        self.Y = [np.multiply(1/logGrad[i], self.Y[i]) for i in range(len(self.Y))]

        for i in range(len(self.Y)):
            self.gradB3 = np.add(self.gradB3, self.Y[i])

        #from y to W3 and h2
        for i in range(len(self.Y)):
            self.gradW3 = np.add(self.gradW3, np.matmul(np.reshape(self.Y[i], (10,1)), np.reshape((self.h2[i]), (1,500))))

        gradB2 = np.zeros(500, float)
        for i in range(len(self.Y)):
            gradB2 = np.add(gradB2, np.matmul(np.transpose(self.W3), np.reshape((self.Y[i]), (10,1))).reshape(500))

        #from h2 to W2 and h1
        self.h2 = [np.multiply(gradB2, np.reshape(np.where(self.h2[i]>0, 1, 0), (1,500))) for i in range(len(self.h2))]

        for i in range(len(self.Y)):
            self.gradB2 = np.add(self.gradB2, self.h2[i])

        for i in range(len(self.Y)):
            self.gradW2 = np.add(self.gradW2, np.matmul(np.reshape(self.h2[i], (500,1)), np.reshape((self.h1[i]), (1,1000))))

        gradB1 = np.zeros(1000, float)
        for i in range(len(self.Y)):
            gradB1 = np.add(gradB1, np.matmul(np.transpose(self.W2), np.reshape((self.h2[i]), (500,1))).reshape(1000))

        #from h1 to W1
        self.h1 = [np.multiply(gradB1, np.reshape(np.where(self.h1[i]>0, 1, 0), (1,1000))) for i in range(len(self.h1))]

        for i in range(len(self.Y)):
            self.gradB1 = np.add(self.gradB1, self.h1[i])

        for i in range(len(self.Y)):
            self.gradW1 = np.add(self.gradW1, np.matmul(np.reshape(self.h1[i], (1000,1)), np.reshape(self.X[i,:], (1,784))))

    def loss_function(self):
        #calculating probabilities for all y
        self.prob = [np.exp(self.Y[i])/np.sum(np.exp(self.Y[i])) for i in range(len(self.Y))]
        entropy = [-np.log10(self.prob[i][self.labels[i]]) for i in range(len(self.prob))] #just take the probability of the true classes
        self.loss = sum(entropy)/len(entropy)
        print(self.count, "   ", self.loss)
        self.count+=1
        
    def step(self):
        self.gradW1 = self.gradW1/len(self.Y)
        self.gradW2 = self.gradW2/len(self.Y)
        self.gradW3 = self.gradW3/len(self.Y)

        self.W1 -= self.stepSize*self.gradW1
        self.W2 -= self.stepSize*self.gradW2
        self.W3 -= self.stepSize*self.gradW3

        self.B1 = self.B1/len(self.Y)
        self.B2 = self.B2/len(self.Y)
        self.B3 = self.B3/len(self.Y)

        self.B1 -= self.stepSize*self.gradB1[0]
        self.B2 -= self.stepSize*self.gradB2[0]
        self.B3 -= self.stepSize*self.gradB3[0]
        
    def clearVariables(self):
        self.Y.clear()
        self.h1.clear()
        self.h2.clear()
        self.prob.clear()

        self.gradW1 = np.zeros((1000,784), float)
        self.gradW2 = np.zeros((500,1000), float)
        self.gradW3 = np.zeros((10,500), float)

        self.gradB1 = np.zeros(1000, float)
        self.gradB2 = np.zeros(500, float)
        self.gradB3 = np.zeros(10, float)

    def test(self, testX):
        #first hidden layer
        h1 = [np.matmul(self.W1, np.transpose(testX[i,:]))+self.B1 for i in range(testX.shape[0])]
        h1 = [np.maximum(0,h1[i]) for i in range(len(h1))]
        
        #second hidden layer
        h2 = [np.matmul(self.W2, h1[i])+self.B2 for i in range(len(h1))]
        h2 = [np.maximum(0,h2[i]) for i in range(len(h2))]
        
        #The result
        Y = [np.matmul(self.W3, h2[i])+self.B3 for i in range(len(h2))]

        return Y

if __name__ == '__main__':
    
    pathToTrainingData = "digit-recognizer/train.csv"

    #getting the training data 
    df = pd.read_csv(pathToTrainingData)

    label = df["label"].to_numpy() #get labels

    df.drop(["label"], axis="columns", inplace=True) #drop the labels, get the image data
    dataset = df.to_numpy()

    #splitting the data into training and validation sets
    trainSet, validationSet, labelTrainSet, labelValSet = train_test_split(dataset, label, train_size=0.8, random_state=42, stratify=label)

    #Training
    model = Model(trainSet[labelTrainSet == 2], labelTrainSet[labelTrainSet == 2])

    while(model.loss > 0.1):

        model.forward() #forward calculation

        model.loss_function() #loss calculation

        model.backward() #back propogation

        model.step() #take the step to the optimum

        model.clearVariables()

    print(model.test(trainSet[0:10]))
    print(labelTrainSet[0:10])

