"""Calculates accuracy and cost of
training, test and critiacal set
run on all the data available in
network/fetch_2D_data"""

import numpy as np
import sys
sys.path.append('network')
sys.path.append('../')
sys.path.append('../../')
from fetch_2D_data import fetch_data
from NN import NeuralNet

X, Y, X_crit, Y_crit= fetch_data()
### LOG-REG CASE
# nn = NeuralNet(X,Y, nodes = [X.shape[1], 2], activations = [None], cost_func='log')
# nn.split_data(frac=0.5, shuffle=True)
# nn.feed_forward(nn.xTrain)
# nn.backpropagation()
# nn.TrainNN(epochs = 100, eta = 0.001, n_print=5)

# 100% ACCURACY MADDAFAKKA
nn = NeuralNet(X,Y, nodes = [X.shape[1],10,2], activations = ['tanh',None],\
                cost_func='log', regularization='l2', lamb=0.01)
nn.split_data(frac=0.5, shuffle=True)
nn.TrainNN(epochs = 200, eta0 = 0.01, n_print=5)

ypred_crit = nn.feed_forward(X_crit, isTraining=False)
critError = nn.cost_function(Y_crit, ypred_crit)
critAcc = nn.accuracy(Y_crit, ypred_crit)

print("Critical error: %g,  Critical accuracy: %g" %(critError, critAcc))
