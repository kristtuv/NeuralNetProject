import numpy as np
import matplotlib.pyplot as plt
from fetch_2D_data import fetch_data
from NN import NeuralNet
from sklearn.utils import shuffle
def plot_regularization():

     #######################################################
    ###############Defining parameters#####################
    #######################################################
    lambs = np.logspace(-4, 0, 8)
    # n_samples =[100, 1000, 4000, 10000]
    n_samples = [40000]

    #######################################################
    ###############Fetching Data###########################
    #######################################################
    X, Y, X_critical, Y_critical = fetch_data()
    X, Y = shuffle(X, Y)
    X_critical, Y_critical = shuffle(X_critical, Y_critical)



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
            nn = NeuralNet( X_train, Y_train, nodes = [X.shape[1],  10, 2],
                    activations = ['sigmoid', None], cost_func='log')
            nn_l1 = NeuralNet( X_train, Y_train, nodes = [X.shape[1],  10, 2],
                    activations = ['sigmoid', None], cost_func='log', regularization='l1', lamb=lamb)
            nn_l2 = NeuralNet( X_train, Y_train, nodes = [X.shape[1],  10, 2],
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

        errfig.savefig('error'+str(sample)+'.png')
        accfig.savefig('accuracy'+str(sample)+'.png')
        critfig.savefig('crit'+str(sample)+'.png')


def check_regularization():
    # np.random.seed(1200)
    # X, Y = fetch_data(5000, -5000, False)
    X, Y, _, _ = fetch_data()
    X, Y = shuffle(X, Y)
    lambs = [0.00001, 0.0001, 0.0008, 0.0009, 0.001, 0.002, 0.01, 0.1, 0.5]
    for i in range(1,2):
        X = X[:i*5000]; Y = Y[:i*5000]
        nn = NeuralNet( X, Y, nodes = [X.shape[1],  10, 2],
                activations = ['sigmoid', None], cost_func='log')

        for lamb in lambs:
            nn_l1 = NeuralNet( X, Y, nodes = [X.shape[1],  10, 2],
                    activations = ['sigmoid', None], cost_func='log', regularization='l1', lamb=lamb)
            nn_l2 = NeuralNet( X, Y, nodes = [X.shape[1],  10, 2],
                    activations = ['sigmoid', None], cost_func='log', regularization='l2', lamb=lamb)
            nn.split_data(frac=0.5, shuffle=True)
            nn_l1.split_data(frac=0.5, shuffle=True)
            nn_l2.split_data(frac=0.5, shuffle=True)
            print("################################")
            print("############### %d #############" % (500*i))
            print("############### %g #############" % (lamb))
            print("################################")
            if lamb==lambs[0]:
                print("#####Normal#####")
                nn.TrainNN(epochs = 4, eta0 = 0.05, n_print=10)
            if lamb >= 0.0001 and lamb < 0.01:

                print("#####L1#####")
                nn_l1.TrainNN(epochs = 400, eta0 = 0.05, n_print=10)

            if lamb >= 0.01:
                print("#####L2#####")
                nn_l2.TrainNN(epochs = 250, eta0 = 0.05, n_print=10)

def best_regularization():

    #####################
    ###Organize data#####
    #####################
    np.random.seed(1200)
    X, Y, X_critical, Y_critical = fetch_data()

    X, Y = shuffle(X, Y)
    X_critical, Y_critical = shuffle(X_critical, Y_critical)
    # X_critical = X_critical[:50000]

    # Y_critical = Y_critical[:50000]
    X = X[:4000]
    Y = Y[:4000]

    #####################
    ###Neural Net #######
    #####################
    nn_l1 = NeuralNet( X, Y, nodes = [X.shape[1], 10,  10, 2],
            activations = ['sigmoid', None], cost_func='log', regularization='l1', lamb=0.002)
    nn_l1.split_data(frac=0.5, shuffle=True)



    # nn_l2 = NeuralNet( X, Y, nodes = [X.shape[1],  10, 2],
    #         activations = ['sigmoid', None], cost_func='log', regularization='l2', lamb=0.1)
    # nn_l2.split_data(frac=0.5, shuffle=True)
    print("#####L1#####")
    nn_l1.TrainNN(epochs = 300, eta0 = 0.05, n_print=10)
    # print("#####L2#####")
    # nn_l2.TrainNN(epochs = 250, eta0 = 0.05, n_print=10)

    #####################
    ####Critical error######
    #####################

    ypred_crit = nn_l1.feed_forward(X_critical, isTraining=False)
    critError = nn_l1.cost_function(Y_critical, ypred_crit)
    critAcc = nn_l1.accuracy(Y_critical, ypred_crit)

    # ypred_crit = nn_l2.feed_forward(X_critical, isTraining=False)
    # critError = nn_l2.cst_function(Y_critical, ypred_crit)
    # critAcc = nn_l2.accuracy(Y_critcal, ypred_crit)

    print("Critical error: %g,  Critical accuracy: %g" %(critError, critAcc))



if __name__=='__main__':
    # best_regularization()
    # check_regularization()
    plot_regularization()
exit()

# #    LOG-REG CASE
# #    nn = NeuralNet(X,Y, nodes = [X.shape[1], 2], activations = [None], cost_func='log')
# #    nn.split_data(frac=0.5, shuffle=True)
# #    nn.feed_forward(nn.xTrain)
# #    nn.backpropagation()
# #    nn.TrainNN(epochs = 100, eta = 0.001, n_print=5)


# #  permutations = []
# #  # permutations = set()
# #  for p in multiset_permutations(activations):
# #      permutations.append(p + [None])
# #      permutations.append(p[:-1] + [None])
# #      permutations.append(p[:-2] + [None])

# #  unique_activations = [list(x) for x in set(tuple(x) for x in permutations)]
# #  unique_activations = sorted(unique_activations, key=len)


#  def architecture(activations = ['sigmoid', 'tanh', 'relu'], n_hidden_layers=2):
#      acts = [[None]]
#      for hl in range(1, n_hidden_layers+1):
#          for activation in activations:
#              acts.append([ activation ]*hl + [None])
#      return acts

#  if __name__ == '__main__':

#      regularizations = ['l1', 'l2']
#      activations = activation_list()
#      learning_rates =  np.logspace(-3, 2, 10)
#      print(activations)


#      exit()

#  cost_functions = ['log', 'mse']
#  learning_rates = [0.001, 0.01, 0.1, 1, 10]
#  nodes = [X.shape[1],



#  # _     _       _     _     _       _
# # | |__ | | __ _| |__ | |__ | | __ _| |__
# # | '_ \| |/ _` | '_ \| '_ \| |/ _` | '_ \
# # | |_) | | (_| | | | | |_) | | (_| | | | |
# # |_.__/|_|\__,_|_| |_|_.__/|_|\__,_|_| |_|





# # X, Y, X_crit, Y_crit= fetch_data()
# # # 100% ACCURACY MADDAFAKKA
# # nn = NeuralNet(X,Y, nodes = [X.shape[1],  10, 2], activations = ['sigmoid', None], cost_func='log', regularization='l2', lamb=0.1)
# # nn.split_data(frac=0.5, shuffle=True)
# # nn.TrainNN(epochs = 100, eta0 = 0.01, n_print=5)

# # # ypred_crit = nn.feed_forward(X_crit, isTraining=False)
# # # critError = nn.cost_function(Y_crit, ypred_crit)
# # # critAcc = nn.accuracy(Y_crit, ypred_crit)

# # # print("Critical errors: %g,  Critical accuracy: %g" %(critError, critAcc))




# # # ### LOG-REG CASE
# # # # nn = NeuralNet(X,Y, nodes = [X.shape[1], 2], activations = [None], cost_func='log')
# # # # nn.split_data(frac=0.5, shuffle=True)
# # # # nn.feed_forward(nn.xTrain)
# # # # nn.backpropagation()
# # # # nn.TrainNN(epochs = 100, eta = 0.001, n_print=5)
