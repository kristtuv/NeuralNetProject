import numpy as np
import matplotlib.pylab as plt
from NN import NeuralNet

def LJ(r):
    return 4*(r**(-12) - r**(-6))

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

nn.TrainNN()

output = nn.feed_forward(nn.xTrain, isTraining=False)
p = np.argsort(nn.xTrain.flatten())
plt.plot(nn.xTrain[p], output[p], label = "After")
plt.legend()
plt.show()
