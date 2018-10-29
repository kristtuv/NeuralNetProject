import numpy as np
from fetch_2D_data import fetch_data
from NN import NeuralNet

X, Y = fetch_data()

nn = NeuralNet(X,Y, nodes = [X.shape[1], 2], activations = [None], cost_func='log')
nn.split_data(frac=0.5, shuffle=True)
#nn.feed_forward(nn.xTrain)
#nn.backpropagation()

nn.TrainNN(epochs = 100, eta = 0.001, n_print=5)
