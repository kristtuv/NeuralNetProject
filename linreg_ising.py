import numpy as np
import scipy.sparse as sp
import scipy.linalg as scl
from cls_reg import LinReg
import matplotlib.pylab as plt

np.random.seed(12)

import warnings
warnings.filterwarnings('ignore')

### define Ising model params
# system size
L = 40

# create 10000 random Ising states
states = np.random.choice([-1, 1], size=(10000,L))

def ising_energies(states, L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J = np.zeros((L, L))
    for i in range(L):
        J[i, (i+1)%L] -= 1.0
    #compute energies
    E = np.einsum('...i,ij,...j->...', states,J,states)

    return E

#calculate Ising energies
energies= ising_energies(states, L)


# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))

# build final data set
energies = energies.reshape(-1, 1)
Data=[states,energies]

n_samples = 400

X = Data[0][:n_samples]
Y = Data[1][:n_samples]
print(Y.shape)

#J = (np.linalg.pinv(X) @ Y).reshape(L, L)
model = LinReg(X, Y)
model.lamb = 0.1
J = model.lasso()
Ypred = X @ J
print(model.MSE(Y, Ypred))
print(model.R2(Y, Ypred))

bias, variance, train_error, test_error = model.bootstrap(1000, model.lasso)
print(bias, variance, train_error, test_error)

mse_train, mse_test, r2_train, r2_test = model.kfold(10, model.lasso)
print(mse_train, mse_test, r2_train, r2_test)

cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
plt.imshow(J.reshape(L, L), **cmap_args)
plt.colorbar()
plt.show()
