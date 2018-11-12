import numpy as np
import matplotlib.pylab as plt
import sys

class NeuralNet():

    def __init__(
        self,
        xData,
        yData,
        nodes=[1, 10, 1],
        activations=['sigmoid', None],
        cost_func='mse',
        regularization=None,
        lamb = 0.0):


        """
        param: xData: Data for training and testing
        param: yData: Reference data for evaluating the model
        param: nodes: Nodes in each layer. First element should
        be the size of one training example and last element depends
        on what kind of output we want. I.e for regression we have 1
        node and for classification we can have several.
        param: activations: activations in the hidden layers of your model.
        There is no activation in the input layer, and the size of activaion
        should be the size of nodes-1. The activation in the output layer
        is normally set to None.
        param: cost_func: Type of cost function to use
        param: regularization: type of regularization
        param: lamb: strength of regularization
        type: xData: ndarray
        type: yData: ndarray
        type: nodes: list
        type: activation: list
        type: cost_func: string
        type: regularization: string
        type: lamb: float
        """
        self.xData = xData
        self.yData = yData
        self.N = xData.shape[0]
        self.cost_func = cost_func
        self.regularization = regularization
        self.lamb = lamb

        if len(nodes) != (len(activations) +1):
            print("Error: invaled lengths of 'nodes' and 'activations.")
            print("Usage: nodes = [input, hidden layers (any no.), output], activations \
                    = activations between layers, len(nodes) = len(activations) + 1. \nExiting...")
            sys.exit(0)

        self.nodes = nodes
        self.activations = activations
        self.nLayers = len(activations)

        self.split_data(folds = 10, frac = 0.3, shuffle=True)
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
            randomize = np.arange(self.N)
            np.random.shuffle(randomize)
            xData = xData[randomize]
            yData = yData[randomize]

        if folds != None:
            xFolds = np.array_split(xData, folds, axis = 0)
            yFolds = np.array_split(yData, folds, axis = 0)

            self.xFolds = xFolds
            self.yFolds = yFolds

        if frac != None:
            nTest = int(np.floor(frac*self.N))
            xTrain = xData[:-nTest]
            xTest = xData[-nTest:]

            yTrain = yData[:-nTest]
            yTest = yData[-nTest:]

            self.xTrain = xTrain ; self.xTest = xTest
            self.yTrain = yTrain ; self.yTest = yTest
            self.nTrain = xTrain.shape[0] ; self.nTest = xTest.shape[0]

    def initialize_weights_biases(self):
        """
        Initializes weights and biases for all layers
        return: None
        """

        self.Weights = {} ; self.Biases = {}
        self.Weights_grad = {} ; self.Biases_grad = {}
        self.Z = {} ; self.A = {} ; self.C = {}

        for i in range(len(self.activations)):

            self.Weights['W'+str(i+1)] = np.random.uniform(-0.1, 0.1, (self.nodes[i], self.nodes[i+1]))
            self.Biases['B'+str(i+1)] = np.random.uniform(-0.1, 0.1, self.nodes[i+1])

            self.Weights_grad['dW'+str(i+1)] = np.zeros_like(self.Weights['W'+str(i+1)])
            self.Biases_grad['dB'+str(i+1)] = np.zeros_like(self.Biases['B'+str(i+1)])


    def activation(self, x, act_func):
        """
        Calculation of the selected
        activation function

        param: x: data
        type: x: ndarray
        param: act_func: activaion function given in init
        type: act_func: string
        return: selected activation
        """


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
        """
        Calculation of derivative of the selected
        activation function

        param: x: data
        type: x: ndarray
        param: act_func: activaion function given in init
        type: act_func: string
        return: derivative of activation function
        """

        if act_func == 'sigmoid':

            return x*(1 - x)

        elif act_func == 'tanh':

            return 1 - x**2

        elif act_func == 'relu':

            return 1*(x >= 0)

        elif act_func == None:
            return 1
        else:
            print("Invalid activation function. Either 'sigmoid', 'tanh', 'relu', or None.\nExiting...")
            sys.exit(0)

    def softmax(self, act):
        """
        calculates the softmax function

        param: act:
        type: act:
        return: softmax function
        """
        # Subtraction of max value for numerical stability
        act_exp = np.exp(act - np.max(act))
        self.act_exp = act_exp

        return act_exp/np.sum(act_exp, axis=1, keepdims=True)

    def cost_function(self, y, ypred):
        """
        Using the cost function defined
        in init

        param: y: correct labels
        type: y: ndarray
        param: y_pred: predicted labels
        type: y_pred: ndarray
        return: selected cost function
        """

        if self.cost_func == 'mse':
            cost =  0.5/y.shape[0]*np.sum((y - ypred)**2)

        if self.cost_func == 'log':
            cost = -0.5/y.shape[0]*np.sum(np.log(ypred[np.arange(ypred.shape[0]), y.flatten()]))

        if self.regularization == 'l2':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(self.Weights[key]**2)

        elif self.regularization == 'l1':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(np.abs(self.Weights[key]))

        return cost


    def cost_function_derivative(self, y, ypred):
        """
        Takes the derivative of the selected cost function

        param: y: correct labels
        type: y: ndarray
        param: y_pred: predicted labels
        type: y_pred: ndarray
        return: costfunction derivative
        """

        if self.cost_func == 'mse':
            return -1.0/y.shape[0]*(y - ypred)

        elif self.cost_func == 'log':
            ypred[np.arange(ypred.shape[0]), y.flatten()] -= 1
            return 1.0/y.shape[0]*ypred

    def accuracy(self, y, ypred):
        """
        Measures the number of correctly
        classified classes
        param: y: correct labels
        type: y: ndarray
        param: y_pred: predicted labels
        type: y_pred: ndarray
        return: accuracy
        """
        cls_pred = np.argmax(ypred, axis=1)
        return 100.0/y.shape[0]*np.sum(cls_pred == y)


    def feed_forward(self, x, isTraining = True):
        """
        Doing the forward propagation

        param: x: Data
        type: x: ndarray
        param: isTraining: Set to false if using a finished
        model for predicting on new data.
        type: isTraining: bool
        return: activation values
        """

        self.A['A0'] = x
        for i in range(self.nLayers):

            z = self.A['A'+str(i)] @ self.Weights['W'+str(i+1)] + self.Biases['B'+str(i+1)]
            a = self.activation(z, self.activations[i])
            self.Z['Z'+str(i+1)] = z
            self.A['A'+str(i+1)] = a

        if self.cost_func == 'log':
            a = self.softmax(a)

        #self.output = a

        if isTraining:
            self.output = a
        else:
            return a



    def backpropagation(self, yTrue = None):
        """
        Function for doing the backpropagation

        param: yTrue: True values of y
        type: yTrue: ndarray
        return: None
        """
        if yTrue is None:
            yTrue = self.yTrain

        for i in range(self.nLayers, 0, -1):

            if i == self.nLayers:
                c = self.cost_function_derivative(yTrue, self.output)
            else:
                c = c @ self.Weights['W'+str(i+1)].T
                c = c * self.activation_derivative(self.A['A'+str(i)], self.activations[i-1])

            grad_w = self.A['A'+str(i-1)].T @ c
            grad_b = np.sum(c, axis= 0)

            self.Weights_grad['dW'+str(i)] = grad_w
            self.Biases_grad['dB'+str(i)] = grad_b

            if self.regularization == 'l2':
                self.Weights['W'+str(i)] -= self.eta*(grad_w + self.lamb*self.Weights['W'+str(i)])

            elif self.regularization == 'l1':
                self.Weights['W'+str(i)] -= self.eta*(grad_w + self.lamb*np.sign(self.Weights['W'+str(i)]))

            else:
                self.Weights['W'+str(i)] -= self.eta*grad_w

            self.Biases['B'+str(i)] -= self.eta*grad_b



    def TrainNN(self, epochs = 1000, batchSize = 200, eta0 = 0.01, n_print = 100):
        """
        Training the network using forward and backward propagation.

        param: epochs: Number of iterations through the entire data set
        type: epochs: int
        param: batchSize: Batch size. Must be between one and the size of the
        full data set
        type: batchSize: int
        param: eta0: learning rate or 'schedule'
        type: eta0: float, string
        param: n_print: how often we print accuracy and error to the terminal
        type: n_print: int
        return: None
        """

        if eta0 == 'schedule':
            t0 = 5 ; t1 = 50
            eta = lambda t : t0/(t + t1)
        else:
            eta = lambda t : eta0

        num_batch = int(self.nTrain/batchSize)

        self.convergence_rate = {'Epoch': [], 'Test Accuracy': []}
        for epoch in range(epochs +1):

            indices = np.random.choice(self.nTrain, self.nTrain, replace=False)

            for b in range(num_batch):

                self.eta = eta(epoch*num_batch+b)

                batch = indices[b*batchSize:(b+1)*batchSize]
                xBatch = self.xTrain[batch]
                yBatch = self.yTrain[batch]

                self.feed_forward(xBatch)
                self.backpropagation(yBatch)

            if epoch == 0 or epoch % n_print == 0:

                ypred_train = self.feed_forward(self.xTrain, isTraining=False)
                ypred_test = self.feed_forward(self.xTest, isTraining=False)
                trainError = self.cost_function(self.yTrain, ypred_train)
                testError = self.cost_function(self.yTest, ypred_test)
                print("Error after %i epochs, Training:  %g, Test:  %g" %(epoch, trainError,testError))

                if self.cost_func == 'log':
                    trainAcc = self.accuracy(self.yTrain, ypred_train)
                    testAcc = self.accuracy(self.yTest, ypred_test)
                    print("Accuracy after %i epochs, Training:   %g %%, Test:   %g %%\n" %(epoch, trainAcc, testAcc))
                    self.convergence_rate['Epoch'].append(epoch)
                    self.convergence_rate['Test Accuracy'].append(testAcc)
                    #print("-"*75)
