import numpy as np

""" Simple classifier for classifying whether a number is larger or smaller than 0.7"""

N = 200
n = 10
p = 2
eta = 1.0
epochs = 10000

xTrain = np.random.uniform(0, 1, (N,1))
yTrain = 1*(xTrain > 0.7)
X = np.c_[np.ones((N, 1)), xTrain]

xTest = np.random.uniform(0, 1, (n,1))
yTest = 1*(xTest > 0.7)
X_Test = np.c_[np.ones((n, 1)), xTest]

beta = np.random.uniform(-0.5, 0.5, (p, 1))

for epoch in range(epochs + 1):

    p = np.exp(X @ beta)/(1 + np.exp(X @ beta))

    dB = -X.T @ (yTrain - p)
    beta -= eta*dB

    if epoch % 100 == 0 or epoch == 0:
        cost = np.sum(yTrain * (X @ beta) - np.log(1 + np.exp(X @ beta)))
        print("Cost after %i epochs: %f" %(epoch, cost))

pTest = np.exp(X_Test @ beta)/(1 + np.exp(X_Test @ beta))
print(xTest[:10].flatten())
print(yTest[:10].flatten())
print(pTest[:10].flatten())
