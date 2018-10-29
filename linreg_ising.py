import numpy as np
import scipy.sparse as sp
import scipy.linalg as scl
from cls_reg import LinReg
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(12)

import warnings
warnings.filterwarnings('ignore')

def stats():

    lambdas = np.logspace(-4, 5,10)
    model = LinReg(X, Y)


    models = []
    for regmethod in ['ols', 'ridge', 'lasso']:

        method = getattr(model, regmethod)

        for lamb in lambdas:

            model.lamb = lamb

            J = method(model.xTrain, model.yTrain)
            Ypred_train = model.xTrain @ J
            Ypred_test = model.xTest @ J

            mse_train = model.MSE(model.yTrain, Ypred_train)
            mse_test = model.MSE(model.yTest, Ypred_test)
            r2_train = model.R2(model.yTrain, Ypred_train)
            r2_test = model.R2(model.yTest, Ypred_test)

            models.append([regmethod, lamb, mse_train, mse_test,\
                    r2_train, r2_test])

            if regmethod == 'ols':
                break

    print("\nMODEL ANALYSIS:")
    print("="*85)
    print(" Method | lambda | MSE Train | MSE Test | R2 Train |  R2 Test |")
    print("-"*85)

    for i in range(len(models)):
        print("%8s|%8g|%11g|%10f|%10f|%10f|" % tuple(models[i]))

    print("-"*85)


    #r2s = np.array([models[i][4:] for i in range(len(models))])
    #plt.semilogx(lambdas, np.tile(r2s[0], (len(lambdas),1)))
    #plt.show()

def boot_stats():

    lambdas = np.logspace(-4, 5,10)
    model = LinReg(X, Y)


    models = []
    print("Bootstrapping models:")
    for regmethod in ['ols', 'ridge', 'lasso']:

        method = getattr(model, regmethod)

        for lamb in lambdas:

            model.lamb = lamb

            bias, variance, mse_train, mse_test, r2_train, r2_test = model.bootstrap(100, method)
            models.append([regmethod, lamb, mse_train, mse_test,\
                    r2_train, r2_test, bias, variance])

            if regmethod == 'ols':
                break

    print("\nMODEL ANALYSIS (BOOTSTRAP):")
    print("="*85)
    print(" Method | lambda | MSE Train | MSE Test | R2 Train |  R2 Test |   Bias   | Variance")
    print("-"*85)

    for i in range(len(models)):
        print("%8s|%8g|%11g|%10f|%10f|%10f|%10f|%10g|" % tuple(models[i]))

    print("-"*85)

def plot_stuff():

    model = LinReg(X,Y)

    fig, axarr = plt.subplots(nrows=2, ncols=3)
    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

    lambdas = [0.0001, 0.01]
    for i in range(len(lambdas)):

        model.lamb = lambdas[i]

        J_ols = model.ols()[1:].reshape(L,L)
        J_ridge = model.ridge()[1:].reshape(L,L)
        J_lasso = model.lasso()[1:].reshape(L,L)

        axarr[i][0].imshow(J_ols,**cmap_args)
        axarr[i][0].set_title('$\\mathrm{OLS}$',fontsize=16)
        axarr[i][0].tick_params(labelsize=16)

        axarr[i][1].imshow(J_ridge,**cmap_args)
        axarr[i][1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lambdas[i]),fontsize=16)
        axarr[i][1].tick_params(labelsize=16)

        im=axarr[i][2].imshow(J_lasso,**cmap_args)
        axarr[i][2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(lambdas[i]),fontsize=16)
        axarr[i][2].tick_params(labelsize=16)

        divider = make_axes_locatable(axarr[i][2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig.colorbar(im, cax=cax)

        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
        cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)

        #fig.subplots_adjust(right=2.0)
    plt.show()


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

#X = np.c_[np.ones(X.shape[0]), X]

if __name__ == "__main__":
    #plot_stuff()
    #boot_stats()
    stats()
