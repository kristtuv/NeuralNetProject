import numpy as np
import sys
sys.path.append('network')
sys.path.append('../')
sys.path.append('../../')
import matplotlib.pylab as plt
from NN import NeuralNet
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

n_samples = 10000

X = Data[0][:n_samples]
Y = Data[1][:n_samples]
#X = np.c_[np.ones(X.shape[0]), X]

def reg():
    """
    Doing the linear regression using
    a neural network
    """

    lambdas = np.logspace(-4, 2, 7)
    #lambdas = [0.01, 0.1]
    n = len(lambdas)
    train_mse = np.zeros((3,n))
    test_mse = np.zeros((3,n))
    regs = [None, 'l1', 'l2']

    for j in range(3):
        for i in range(n):
            nn = NeuralNet(X,Y, nodes=[X.shape[1], 1], activations=[None], regularization=regs[j], lamb=lambdas[i])
            nn.TrainNN(epochs = 1000, batchSize = 200, eta0 = 0.01, n_print = 10)

            ypred_test = nn.feed_forward(nn.xTest, isTraining=False)
            mse_test = nn.cost_function(nn.yTest, ypred_test)

            test_mse[j,i] = mse_test

            if j == 0:
                test_mse[0].fill(test_mse[0,0])
                break


    plt.semilogx(lambdas, test_mse.T)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.title("MSE on Test Set, %i samples" %(n_samples))
    plt.legend(['No Reg', 'L1', 'L2'])
    plt.grid(True)
    plt.ylim([0, 25])
    plt.show()


def plot():
    """
    Ploting the matrix of the coupling constants
    """

    n = [400, 10000]
    fig, axarr = plt.subplots(nrows=2, ncols=3)
    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

    for i in range(len(n)):
        n_samples = n[i]

        X = Data[0][:n_samples]
        Y = Data[1][:n_samples]

        regs = [None, 'l2','l1']
        Js = np.zeros((3, L**2))

        for j in range(3):

            nn = NeuralNet(X,Y, nodes=[X.shape[1], 1], activations=[None], regularization=regs[j], lamb=0.1)
            nn.TrainNN(epochs = 500, batchSize = 200, eta0 = 0.01, n_print = 50)
            Js[j] = nn.Weights['W1'].flatten()

        J_ols = Js[0].reshape(L,L)
        J_ridge = Js[1].reshape(L,L)
        J_lasso = Js[2].reshape(L,L)

        axarr[i][0].imshow(J_ols,**cmap_args)
        axarr[i][0].set_title('$\\mathrm{OLS}$',fontsize=16)
        axarr[i][0].tick_params(labelsize=16)

        axarr[i][1].imshow(J_ridge,**cmap_args)
        axarr[i][1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(0.1),fontsize=16)
        axarr[i][1].tick_params(labelsize=16)

        im=axarr[i][2].imshow(J_lasso,**cmap_args)
        axarr[i][2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(0.1),fontsize=16)
        axarr[i][2].tick_params(labelsize=16)

        divider = make_axes_locatable(axarr[i][2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig.colorbar(im, cax=cax)

        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
        cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)

        #fig.subplots_adjust(right=2.0)
    plt.show()


#plt.imshow(nn.Weights['W1'].reshape(L,L), cmap='seismic', vmin=-1, vmax=1)
#plt.show()

if __name__=='__main__':
    reg()
    #plot()
