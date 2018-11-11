"""Needs description"""
import sys
sys.append('network')
sys.append('../')
sys.append('../../')

import numpy as np
import matplotlib.pylab as plt
from NN import NeuralNet
import numpy as np
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

def LJ(r):
    return 4*(r**(-12) - r**(-6))

def ising_create_states(n_states, lattice_size):
    return np.random.choice([-1, 1], size=(n_states,lattice_size))

def ising_create_energies(states, lattice_size):
    J=np.zeros((lattice_size,lattice_size),)
    for i in range(lattice_size):
        J[i,(i+1)%lattice_size]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E



def testing_of_nn():
    N = 2000
    xData = np.random.uniform(1.0, 2.0, (N,1))
    #xData = np.random.uniform(-1.0, 1.0, (N,1))
    yData = LJ(xData)
    #yData = xData**3

    xPlot = np.linspace(1.0, 2.0, 1000)
    #xPlot = np.linspace(-1.0, 1.0, 1000)
    yPlot = LJ(xPlot)
    #yPlot = xPlot**3

    nn = NeuralNet(xData, yData)
    #nn.feed_forward(nn.xTrain)
    #nn.backpropagation()

    plt.plot(xPlot, yPlot, label="Exact")
    #plt.plot(nn.xTrain, nn.output, label= "Before")

    nn.TrainNN(epochs = 400000, eta = 0.1, n_print = 1000)

    output = nn.feed_forward(nn.xTrain, isTraining=False)
    p = np.argsort(nn.xTrain.flatten())
    plt.plot(nn.xTrain[p], output[p], label = "After")
    plt.legend()
    plt.show()


if __name__=='__main__':

    np.random.seed(12)
    L = 40
    n_states = 10000
    states = ising_create_states(n_states, L)
    energies=ising_create_energies(states,L).reshape(-1, 1)
    states=np.einsum('...i,...j->...ij', states, states)
    shape=states.shape
    states=states.reshape((shape[0],shape[1]*shape[2]))
    #nn = NeuralNet(states, energies,nodes=[1600, 1], activations=[None], regularization='l2', lamb = 10000.0)
    nn = NeuralNet(states, energies,nodes=[1600, 1], activations=[None])
    nn.TrainNN(epochs=101)
    output = nn.feed_forward(nn.xTrain, isTraining=False)
    #print(output[:10])
    #print(nn.yTrain[:10])

    J = nn.Weights['W1'].reshape(40, 40)

    plt.imshow(J, cmap = 'seismic', vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.show()



    #testing_of_nn()
    #p = np.argsort(nn.xTrain.flatten())
    #plt.plot(nn.xTrain[p], output[p], label = "After")
    #plt.legend()
    #plt.show()



    # build final data set
    # Data=[states,energies]

    # # define number of samples
    # n_samples=400
    # # define train and test data sets
    # X_train=Data[0][:n_samples]
    # Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
    # X_test=Data[0][n_samples:3*n_samples//2]
    # Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
    # print(Y_test)
    # print(states.shape)
