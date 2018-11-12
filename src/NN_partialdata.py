"""
A brute force program for testing different architectures
in neural networks.

Best_architecture writes to csv files a list of sorted
architectures by critical accuracy. Remember to change
the name of csv file, as it will overwrite the previous one.
Change values under 'Defining parameters'
to change type of architecture.

Plot_architcures plots accuracy against differnt
values of lambda.
"""

import numpy as np
import sys
sys.path.append('network')
sys.path.append('../')
import matplotlib.pyplot as plt
from fetch_2D_data import fetch_data
from NN import NeuralNet
from sklearn.utils import shuffle
import pandas as pd
from time import time
def plot_convergence():
    """
    Ploting the convergence rate of the 6 best models
    calculated in best_architectures. 
    """

    ########################################################
    ################Fetching Data###########################
    ########################################################
    sample = 10000
    X, Y, X_critical, Y_critical = fetch_data()
    X, Y = shuffle(X, Y)
    X_critical, Y_critical = shuffle(X_critical, Y_critical)

    X_train = X[:sample]; Y_train = Y[:sample]
    X_test = X[sample:2*sample]; Y_test = Y[sample:2*sample]
    X_crit = X_critical[:sample]; Y_crit = Y_critical[:sample]


    #######################################################
    ###############Defining parameters#####################
    #######################################################

    best_architectures = [['tanh', None], ['tanh', None], ['tanh', None],
            ['tanh','tanh', None], ['tanh', 'tanh', 'tanh', None],
            ['sigmoid', 'sigmoid', None]]
    nodes = [[X.shape[1], 100, 2], [X.shape[1], 50, 2], [X.shape[1], 10, 2],
            [X.shape[1], 100, 100, 2], [X.shape[1], 100, 100, 100, 2],
            [X.shape[1], 100, 100, 2]]


    regularizations = ['l2', 'l2', 'l2', 'l2', 'l1', None]
    lambdas = [0.1, 0.1, 0.1, 0.1, 0.001, 0.01]

    df = pd.DataFrame()
    convergence_dict = {}
    walltime_list = []
    i = 0
    for activation, lamb, node, reg in zip(best_architectures, lambdas, nodes, regularizations):

        nn = NeuralNet(
                    X_train,
                    Y_train,
                    nodes = node,
                    activations = activation,
                    cost_func='log',
                    regularization=reg,
                    lamb=lamb)

        start = time()
        nn.TrainNN(epochs = 100, n_print=1)
        stop = time()
        convergence = nn.convergence_rate
        walltime = (stop - start)
        convergence_dict[activation[0]*(len(activation)-1)+str(node[1])] = convergence['Test Accuracy']
        walltime_list.append(walltime)
        i += 1
    fig, ax = plt.subplots()
    for key in convergence_dict:
        ax.plot(np.arange(101), convergence_dict[key], label=key)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    plt.legend()
    # plt.savefig('../plots/convergence_rate.png')

    figb, bx = plt.subplots()
    names = list(convergence_dict.keys())
    bx.bar(np.arange(len(walltime_list)), walltime_list, tick_label=names)
    bx.set_ylabel('Walltime [s]')
    plt.xticks(rotation=-20)
    plt.savefig('../plots/walltime.png')
    plt.tight_layout()
    plt.show()
def best_architecture():
    """
    Finding best architectures for neural network
    """

    #######################################################
    ###############Defining parameters#####################
    #######################################################
    # lambdas = np.logspace(-4, 0, 5)
    lambdas = [0.01]
    nodes = [5]
    regularizations = [None]
    n_samples = [10, 20]
    activation_functions = [['relu', None]]

    df = pd.DataFrame()

    ########################################################
    ################Fetching Data###########################
    ########################################################
    X, Y, X_critical, Y_critical = fetch_data()
    X, Y = shuffle(X, Y)
    X_critical, Y_critical = shuffle(X_critical, Y_critical)

    for sample in n_samples:
        print(sample)
        X_train = X[:sample]; Y_train = Y[:sample]
        X_test = X[sample:2*sample]; Y_test = Y[sample:2*sample]
        X_crit = X_critical[:sample]; Y_crit = Y_critical[:sample]
        for reg in regularizations:
            print(reg)
            for activation in activation_functions:
                print(activation)
                for n in nodes:
                    print(n)
                    node = [X.shape[1], 2]
                    node[1:1] = [n]*(len(activation)-1)
                    print(node)
                    for lamb in lambdas:
                        print(lamb)
                        nn = NeuralNet(
                                    X_train,
                                    Y_train,
                                    nodes = node,
                                    activations = activation,
                                    cost_func='log',
                                    regularization=reg,
                                    lamb=lamb)
                        nn.TrainNN(epochs = 100)

                        ypred_train = nn.feed_forward(X_train, isTraining=False)
                        ypred_test = nn.feed_forward(X_test, isTraining=False)
                        ypred_crit = nn.feed_forward(X_crit, isTraining=False)

                        df = df.append({
                                'Sample size': sample,
                                'Lambda': lamb,
                                'Regularization': reg,
                                'Nodes': n,
                                'Activation': (len(activation)-1)*activation[0],
                                'Train error': nn.cost_function(Y_train, ypred_train),
                                'Test error':  nn.cost_function(Y_test, ypred_test),
                                'Critical error': nn.cost_function(Y_crit, ypred_crit),
                                'Train accuracy':nn.accuracy(Y_train, ypred_train),
                                'Test accuracy': nn.accuracy(Y_test, ypred_test),
                                'Critical accuracy': nn.accuracy(Y_crit, ypred_crit)
                                }, ignore_index=True)
    df.to_csv('best_architecture.csv', index_label='Index')


def plot_regularization():
    """
    Ploting different models as a function of regularization strength
    """
    #######################################################
    ###############Defining parameters#####################
    #######################################################
    lambs = np.logspace(-4, 0, 5)
    # n_samples =[100, 1000, 4000, 10000]
    n_samples = [3000]
    # nodes = [10]
    nodes=[50]
    #######################################################
    ###############Fetching Data###########################
    #######################################################
    X, Y, X_critical, Y_critical = fetch_data()
    X, Y = shuffle(X, Y)
    X_critical, Y_critical = shuffle(X_critical, Y_critical)

    for node in nodes:
        for sample in n_samples:
            X_train = X[:sample]; Y_train = Y[:sample]
            X_test = X[sample:2*sample]; Y_test = Y[sample:2*sample]
            X_crit = X_critical[:sample]; Y_crit = Y_critical[:sample]


            #######################################################
            ###########Dictionaries for ploting####################
            #######################################################
            errors = {'ols': {'Train': [], 'Test': [], 'Crit': []},
                      'l1':  {'Train': [], 'Test': [], 'Crit': []},
                      'l2':  {'Train': [], 'Test': [], 'Crit': []}}


            accuracies = {'ols': {'Train': [], 'Test': [], 'Crit': []},
                      'l1':  {'Train': [], 'Test': [], 'Crit': []},
                      'l2':  {'Train': [], 'Test': [], 'Crit': []}}


            for lamb in lambs:
                #######################################################
                ###########Initializing networks#######################
                #######################################################
                nn = NeuralNet( X_train, Y_train, nodes = [X.shape[1],  node, 2],
                        activations = ['sigmoid', None], cost_func='log')
                nn_l1 = NeuralNet( X_train, Y_train, nodes = [X.shape[1],  node, 2],
                        activations = ['sigmoid', None], cost_func='log', regularization='l1', lamb=lamb)
                nn_l2 = NeuralNet( X_train, Y_train, nodes = [X.shape[1],  node, 2],
                        activations = ['sigmoid', None], cost_func='log', regularization='l2', lamb=lamb)

                #######################################################
                ###########Spliting data#######################
                #######################################################
                nn.split_data(frac=0.5, shuffle=True)
                nn_l1.split_data(frac=0.5, shuffle=True)
                nn_l2.split_data(frac=0.5, shuffle=True)

                #######################################################
                ###########Training network#######################
                #######################################################

                nn.TrainNN(epochs = 250, eta0 = 0.05, n_print=250)
                nn_l1.TrainNN(epochs = 250, eta0 = 0.05, n_print=250)
                nn_l2.TrainNN(epochs = 250, eta0 = 0.05, n_print=250)


                #######################################################
                ###########Error and accuracies ols#######################
                #######################################################
                ypred_train = nn.feed_forward(X_train, isTraining=False)
                ypred_test = nn.feed_forward(X_test, isTraining=False)
                ypred_crit = nn.feed_forward(X_crit, isTraining=False)
                errors['ols']['Train'].append(nn.cost_function(Y_train, ypred_train))
                errors['ols']['Test'].append(nn.cost_function(Y_test, ypred_test))
                errors['ols']['Crit'].append(nn.cost_function(Y_crit, ypred_crit))
                accuracies['ols']['Train'].append(nn.accuracy(Y_train, ypred_train))
                accuracies['ols']['Test'].append(nn.accuracy(Y_test, ypred_test))
                accuracies['ols']['Crit'].append(nn.accuracy(Y_crit, ypred_crit))

                #######################################################
                ###########Error and accuracies l1#######################
                #######################################################
                ypred_train = nn_l1.feed_forward(X_train, isTraining=False)
                ypred_test = nn_l1.feed_forward(X_test, isTraining=False)
                ypred_crit = nn_l1.feed_forward(X_crit, isTraining=False)
                errors['l1']['Train'].append(nn_l1.cost_function(Y_train, ypred_train))
                errors['l1']['Test'].append(nn_l1.cost_function(Y_test, ypred_test))
                errors['l1']['Crit'].append(nn_l1.cost_function(Y_crit, ypred_crit))
                accuracies['l1']['Train'].append(nn_l1.accuracy(Y_train, ypred_train))
                accuracies['l1']['Test'].append(nn_l1.accuracy(Y_test, ypred_test))
                accuracies['l1']['Crit'].append(nn_l1.accuracy(Y_crit, ypred_crit))

                #######################################################
                ###########Error and accuracies l2#######################
                #######################################################
                ypred_train = nn_l2.feed_forward(X_train, isTraining=False)
                ypred_test = nn_l2.feed_forward(X_test, isTraining=False)
                ypred_crit = nn_l2.feed_forward(X_crit, isTraining=False)
                errors['l2']['Train'].append(nn_l2.cost_function(Y_train, ypred_train))
                errors['l2']['Test'].append(nn_l2.cost_function(Y_test, ypred_test))
                errors['l2']['Crit'].append(nn_l2.cost_function(Y_crit, ypred_crit))
                accuracies['l2']['Train'].append(nn_l2.accuracy(Y_train, ypred_train))
                accuracies['l2']['Test'].append(nn_l2.accuracy(Y_test, ypred_test))
                accuracies['l2']['Crit'].append(nn_l2.accuracy(Y_crit, ypred_crit))


            datasetnames = ['Train', 'Test']
            errfig, errax = plt.subplots()
            accfig, accax = plt.subplots()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            linestyles = ['-', '--']
            for i, key in enumerate(accuracies):
                for j, name in enumerate(datasetnames):
                    errax.set_xlabel(r'$\lambda$')
                    errax.set_ylabel('Error')
                    errax.semilogx(lambs[:-2], errors[str(key)][str(name)][:-2],
                            color=colors[i], linestyle=linestyles[j], label=str(key).capitalize()+'_'+str(name))
                    errax.legend()



                    accax.set_xlabel(r'$\lambda$')
                    accax.set_ylabel('Accuracy')
                    accax.semilogx(lambs, accuracies[str(key)][str(name)],
                            color=colors[i], linestyle=linestyles[j], label=str(key).capitalize()+'_'+str(name))
                    accax.legend()




            critfig, critax = plt.subplots()
            critax.set_xlabel(r'$\lambda$')
            critax.set_ylabel('Accuracy')
            for i, key in enumerate(accuracies):
                critax.semilogx(lambs, accuracies[str(key)]['Crit'],
                     label=str(key).capitalize()+'_Crit')
                critax.legend()
            plt.show()
            # errfig.savefig('error'+str(sample)+'.png')
            # accfig.savefig('accuracy'+str(sample)+'.png')
            # critfig.savefig('crit'+str(sample)+'.png')




if __name__=='__main__':
    #best_architecture()
    #plot_regularization()
    plot_convergence()
