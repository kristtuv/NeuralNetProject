import numpy as np
import sys
sys.path.append('network')
sys.path.append('../')
sys.path.append('../../')
import matplotlib.pylab as plt
import plotparams

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return x*(x>0)

x = np.linspace(-7, 7, 1000)

for act in [sigmoid, tanh, ReLU]:

    plt.plot(x, act(x))
    plt.title("%s activation function" %(act.__name__.capitalize()))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig("plots/%s.png" %(act.__name__.lower()))
    #plt.grid(True)
    plt.show()
