import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numba import jit, cuda

class Model:

    def __init__(self, inputs, labels):
        self.labels = labels

        self.X = inputs
        self.Y = []

        self.loss = 0
        self.stepSize = 1e-5

        self.W1 = np.random.rand(20,784)*1e-4
        self.W2 = np.random.rand(30,20)*1e-4
        self.W3 = np.random.rand(10,30)*1e-4

        self.B1 = np.zeros((20))
        self.B2 = np.zeros((30))
        self.B3 = np.zeros((10))

        self.h2 = []
        self.h1 = []

        self.gradW1 = np.zeros((20,784))
        self.gradW2 = np.zeros((30,20))
        self.gradW3 = np.zeros((10,30))

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
        logGrad = [-1/np.multiply(np.log(10), self.prob[i]) for i in range(len(self.prob))]
        self.Y = [np.where(self.prob[i]==self.prob[i][self.labels[i]], self.prob[i][self.labels[i]], 0)\
            -self.prob[i][self.labels[i]]*self.prob[i] for i in range(len(self.prob))]
        self.Y = [np.multiply(logGrad[i], self.Y[i]) for i in range(len(self.Y))]

        #from y to W3 and h2
        self.gradW3 = sum([np.matmul(np.reshape(self.Y[i], (10,1)), np.reshape((self.h2[i]), (1,30))) for i in range(len(self.Y))])
        gradh2 = sum([np.matmul(np.transpose(self.W3), np.reshape((self.Y[i]), (10,1))) for i in range(len(self.Y))])

        #from h2 to W2 and h1
        self.h2 = [np.multiply(gradh2, np.reshape(np.where(self.h2[i]>0, 1, 0), (30,1))) for i in range(len(self.h2))]
        self.gradW2 = sum([np.matmul(np.reshape(self.h2[i], (30,1)), np.reshape((self.h1[i]), (1,20))) for i in range(len(self.h2))])
        gradh1 = sum([np.matmul(np.transpose(self.W2), np.reshape((self.h2[i]), (30,1))) for i in range(len(self.h2))])

        #from h1 to W1
        self.h1 = [np.multiply(gradh1, np.reshape(np.where(self.h1[i]>0, 1, 0), (20,1))) for i in range(len(self.h1))]
        self.gradW1 = sum([np.matmul(np.reshape(self.h1[i], (20,1)), np.reshape(self.X[i,:], (1,784))) for i in range(len(self.h1))])

    def loss_function(self):
        #calculating probabilities for all y
        self.prob = [np.exp(self.Y[i])/np.sum(np.exp(self.Y[i])) for i in range(len(self.Y))]
        print(self.prob[1], "   ", self.labels[1])
        entropy = -np.log10(self.prob[:][self.labels[i]]) #just take the probability of the true class
        self.loss = sum(entropy)/len(entropy)
        print(self.count, "   ", self.loss)
        self.count+=1
        
    def step(self):
        print(self.gradW1)
        self.W1 -= self.stepSize*self.gradW1
        self.W2 -= self.stepSize*self.gradW2
        self.W2 -= self.stepSize*self.gradW2
        

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
    model = Model(trainSet, labelTrainSet)

    iteration = 1000
    for i in range(iteration):

        model.forward() #forward calculation

        model.loss_function() #loss calculation

        model.backward() #back propogation

        model.step() #take the step to the optimum


