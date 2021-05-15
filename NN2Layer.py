import numpy as np


class NN2Layer:
    def __init__(self, input_dim=1, first_layer_dim=1, second_layer_dim=1, output_dim=1,
                 step_size=0.1, regularization_const=1e-5):

        self.stepSize = step_size
        self.alfa = regularization_const

        self.m = 0
        self.n = input_dim
        self.d1 = first_layer_dim
        self.d2 = second_layer_dim
        self.K = output_dim

        self.W1 = np.random.randn(self.d1, self.n)*np.sqrt(2/self.n)
        self.W2 = np.random.randn(self.d2, self.d1)*np.sqrt(2/self.d1)
        self.W3 = np.random.randn(self.K, self.d2)*np.sqrt(2/self.d2)

        self.h2 = np.zeros(self.d2)
        self.h1 = np.zeros(self.d1)

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
        prediction = np.minimum(np.divide(tmp, np.sum(tmp, axis=0)), 0.9999)

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
        self.gradW3 = np.add(self.gradW3, np.multiply(self.alfa, np.where(self.W3 > 0, 1, -1)))

        # from outputs to h2
        grad_h2 = np.matmul(np.transpose(self.W3), grad_y)
        grad_h2 = np.multiply(np.where(self.h2 > 0, 1, 0), grad_h2)

        # from h2 to W2
        self.gradW2 = np.moveaxis(np.broadcast_to(grad_h2, (self.d1, self.d2, self.m)), 0, 1)
        self.gradW2 = np.mean(np.multiply(self.gradW2, np.broadcast_to(self.h1, (self.d2, self.d1, self.m))), axis=-1)
        self.gradW2 = np.add(self.gradW2, np.multiply(self.alfa, np.where(self.W2 > 0, 1, -1)))

        # from h2 to h1
        grad_h1 = np.matmul(np.transpose(self.W2), grad_h2)
        grad_h1 = np.multiply(np.where(self.h1 > 0, 1, 0), grad_h1)

        # from h1 to W1
        self.gradW1 = np.moveaxis(np.broadcast_to(grad_h1, (self.n, self.d1, self.m)), 0, 1)
        self.gradW1 = np.mean(np.multiply(self.gradW1, np.broadcast_to(x, (self.d1, self.n, self.m))), axis=-1)
        self.gradW1 = np.add(self.gradW1, np.multiply(self.alfa, np.where(self.W1 > 0, 1, -1)))

    def step(self):
        self.W1 -= self.stepSize*self.gradW1
        self.W2 -= self.stepSize*self.gradW2
        self.W3 -= self.stepSize*self.gradW3
