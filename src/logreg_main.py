"""Program for calculating logistic
regression evalutated on two
dimensional ising model"""

import sys
sys.path.append('network')
sys.path.append('../')
sys.path.append('../../')

import numpy as np
from logreg import LogReg_Ising
import time
import matplotlib.pylab as plt

def check_batch():
    """
    Finding the impact of batch sizes and learining
    rate on the cost and acuracy
    """
    num_batches = [1, 100, 500, 1000, 65000]
    rates = [0.001, 0.01, 0.1]
    n = len(num_batches)
    r = len(rates)

    sizes = np.zeros((n,r))
    train_accs = np.zeros((n,r))
    test_accs = np.zeros((n,r))
    train_costs = np.zeros((n,r))
    test_costs = np.zeros((n,r))
    times = np.zeros((n,r))

    model = LogReg_Ising()
    J = model.J.copy()         # Save the initialization

    for i in range(n):
        for j in range(r):

            model.J = J.copy()      # Use the same initialization

            t_start = time.clock()

            model.optimize(m = num_batches[i], epochs = 80, eta = rates[j])

            t_stop = time.clock()
            times[i,j] = (t_stop - t_start)

            sizes[i,j] = model.batchSize
            train_accs[i,j] = model.train_accuracy
            test_accs[i,j] = model.test_accuracy
            train_costs[i,j] = model.train_cost
            test_costs[i,j] = model.test_cost

            print("\n\n")


    print("Batch size|  L.rate | Train Cost| Test Cost | Train Acc | Test Acc |  Wall Time")
    print("-"*80)
    for i in range(n):
        for j in range(r):
            print(" %7g  | %7g |  %7g  |  %7g  |  %7g | %7g | %7g" %(sizes[i,j], \
                    rates[j], train_costs[i,j], test_costs[i,j], train_accs[i,j], \
                    test_accs[i,j], times[i,j]))
        print("-"*80)

def check_reg():
    """
    Checking impact of regularization in logistic regression.
    """

    lambdas = np.logspace(-5, 4, 10)
    regs = ['l2', 'l1']
    n = len(lambdas)
    train_accs = np.zeros((n, 2))
    train_costs = np.zeros((n, 2))
    test_accs = np.zeros((n, 2))
    test_costs = np.zeros((n, 2))

    model = LogReg_Ising()
    J = model.J.copy()

    for i in range(n):
        for j in range(2):

            model.J = J.copy()
            model.optimize(m = 1000, epochs = 80, regularization=regs[j], lamb = lambdas[i])

            train_accs[i,j] = model.train_accuracy
            train_costs[i,j] = model.train_cost
            test_accs[i,j] = model.test_accuracy
            test_costs[i,j] = model.test_cost

            print("\n\n")


    plt.semilogx(lambdas, train_accs[:,0], color ='#1f77b4', label="L2 Train")
    plt.semilogx(lambdas, train_accs[:,1], color ='#ff7f0e',label="L1 Train")
    plt.semilogx(lambdas, test_accs[:,0], '#1f77b4',linestyle='--',label="L2 Test")
    plt.semilogx(lambdas, test_accs[:,1], '#ff7f0e',linestyle='--',label="L1 Test")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.semilogx(lambdas, train_costs[:,0], color='#1f77b4',label="L2 Train")
    plt.semilogx(lambdas, train_costs[:,1], color='#ff7f0e',label="L1 Train")
    plt.semilogx(lambdas, test_costs[:,0], color='#1f77b4',linestyle='--',label="L2 Test")
    plt.semilogx(lambdas, test_costs[:,1], color='#ff7f0e',linestyle='--',label="L1 Test")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.show()


def check_best():
    """
    Finding the best logistic regression model
    """

    model = LogReg_Ising()
    model.optimize(m = 500, epochs = 200, eta = 0.001)

    logit_crit = model.X_critical @ model.J
    p_crit = 1.0/(1.0+np.exp(-logit_crit))
    accuracy_crit = np.sum((p_crit.flatten() > 0.5) == model.Y_critical)/model.X_critical.shape[0]

    print("Critical accuracy: %f %%" %(accuracy_crit))

if __name__ == '__main__':
    #check_reg()
    #check_batch()
    check_best()
