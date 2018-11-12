"""
Neural network calcualting logistic regression
"""
import sys
sys.path.append('network')
sys.path.append('../')
sys.path.append('../../')
import numpy as np
from fetch_2D_data import fetch_data
from NN import NeuralNet

X, Y, X_crit, Y_crit= fetch_data()

nn = NeuralNet(X,Y,nodes=[X.shape[1], 2], activations=[None], cost_func='log')
nn.TrainNN(epochs = 200, batchSize = 130, eta0 = 0.001, n_print=20)

ypred_crit = nn.feed_forward(X_crit, isTraining=False)
critError = nn.cost_function(Y_crit, ypred_crit)
critAcc = nn.accuracy(Y_crit, ypred_crit)

print("Critical error: %g,  Critical accuracy: %g" %(critError, critAcc))
