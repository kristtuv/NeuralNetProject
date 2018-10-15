import numpy as np
import matplotlib.pylab as plt
import sys

class NeuralNet():

    def __init__(self, xData, yData, nodes=[1, 10, 1], activations=['sigmoid', None]):

        self.xData = xData
        self.yData = yData

        if len(nodes) != (len(activations) +1):
            print("Error: invaled lengths of 'nodes' and 'activations.")
            print("Usage: nodes = [input, hidden layers (any no.), output], activations \
                    = activations between layers, len(nodes) = len(activations) + 1. \nExiting...")
            sys.exit(0)

        self.nodes = nodes
        self.activations = activations
        self.nLayers = len(activations)

        self.split_data(folds = 10, frac = 0.3)
        self.initialize_weights_biases()

    def split_data(self, folds = None, frac = None, shuffle = False):
        """
        Splits the data into training and test. Give either frac or folds

        param: folds: Number of folds
        param: frac: Fraction of data to be test data
        param: shuffle: If True: shuffles the design matrix
        type: folds: int
        type: frac: float
        type: shuffle: Bool
        return: None
        """

        if folds == None and frac == None:
            print("Error: No split info received, give either no. folds or fraction.")
            sys.exit(0)

        xData = self.xData
        yData = self.yData

        if shuffle:
            randomize = np.arange(xData.shape[0])
            np.random.shuffle(randomize)
            xData = xData[randomize]
            yData = yData[randomize]

        if folds != None:
            xFolds = np.array_split(xData, folds, axis = 0)
            yFolds = np.array_split(yData, folds, axis = 0)

            self.xFolds = xFolds
            self.yFolds = yFolds

        if frac != None:
            nTest = int(np.floor(frac*xData.shape[0]))
            xTrain = xData[:-nTest]
            xTest = xData[-nTest:]

            yTrain = yData[:-nTest]
            yTest = yData[-nTest:]

            self.xTrain = xTrain ; self.xTest = xTest
            self.yTrain = yTrain ; self.yTest = yTest
            self.nTrain = xTrain.shape[0] ; self.nTest = xTest.shape[0]

    def initialize_weights_biases(self):

        self.Weights = {} ; self.Biases = {}
        self.Weights_grad = {} ; self.Biases_grad = {}
        self.Z = {} ; self.A = {} ; self.C = {}
        self.A['A0'] = self.xTrain

        for i in range(len(self.activations)):

            self.Weights['W'+str(i+1)] = np.random.uniform(-0.1, 0.1, (self.nodes[i], self.nodes[i+1]))
            self.Biases['B'+str(i+1)] = np.random.uniform(-0.1, 0.1, self.nodes[i+1])

            self.Weights_grad['dW'+str(i+1)] = np.zeros_like(self.Weights['W'+str(i+1)])
            self.Biases_grad['dB'+str(i+1)] = np.zeros_like(self.Biases['B'+str(i+1)])


    def activation(self, x, act_func):

        if act_func == 'sigmoid':

            return 1.0/(1.0 + np.exp(-x))

        elif act_func == 'tanh':

            return np.tanh(x)

        elif act_func == 'relu':

            return x * (x >= 0)

        elif act_func == None:
            return x

        else:
            print("Invalid activation function. Either 'sigmoid', 'tanh', 'relu', or None.\nExiting...")
            sys.exit(0)

    def activation_derivative(self, x, act_func):

        if act_func == 'sigmoid':

            return x*(1 - x)

        elif act_func == 'tanh':

            return 1 - x**2

        elif act_func == 'relu':

            return 1*(x >= 0)

        elif act_func == None:
            return x
        else:
            print("Invalid activation function. Either 'sigmoid', 'tanh', 'relu', or None.\nExiting...")
            sys.exit(0)


    def feed_forward(self, x):

        for i in range(self.nLayers):

            z = self.A['A'+str(i)] @ self.Weights['W'+str(i+1)] + self.Biases['B'+str(i+1)]
            a = self.activation(z, self.activations[i])
            self.Z['Z'+str(i+1)] = z
            self.A['A'+str(i+1)] = a

        self.output = a


    def backpropagation(self):

        error_out = self.yTrain - self.output
        cost = 1.0/self.nTrain*(error_out)**2

        for i in range(self.nLayers, 0, -1):

            if i == self.nLayers:
                c = -2/self.nTrain*error_out
            else:
                c = c @ self.Weights['W'+str(i)]

            c = c * self.activation_derivative(self.A['A'+str(i)], self.activations[i-1])
            grad_w = self.A['A'+str(i-1)].T @ c
            grad_b = np.sum(c, axis= 0)

            self.Weights_grad['dW'+str(i)] = grad_w
            self.Biases_grad['dB'+str(i)] = grad_b

            self.Weights['W'+str(i)] -= 0.001*grad_w
            self.Biases['B'+str(i)] -= 0.001*grad_b

    def TrainNN(self):

        self.feed_forward(self.xTrain)
        print(1.0/self.nTrain*np.sum((self.yTrain - self.output)**2))

        for i in range(100000):
            self.feed_forward(self.xTrain)
            self.backpropagation()
        print(1.0/self.nTrain*np.sum((self.yTrain - self.output)**2))
