import numpy as np
from fetch_2D_data import fetch_data
from NN import NeuralNet
from sklearn.utils import shuffle
def check_regularization():
    # X, Y = fetch_data(5000, -5000, False)
    X, Y, _, _ = fetch_data()
    X, Y = shuffle(X, Y)
    X = X[:5000]; Y = Y[:5000]
    
    nn = NeuralNet( X, Y, nodes = [X.shape[1],  10, 2], 
            activations = ['sigmoid', None], cost_func='log', lamb=0.1)
    nn_l1 = NeuralNet( X, Y, nodes = [X.shape[1],  10, 2], 
            activations = ['sigmoid', None], cost_func='log', regularization='l1', lamb=0.1)
    nn_l2 = NeuralNet( X, Y, nodes = [X.shape[1],  10, 2], 
            activations = ['sigmoid', None], cost_func='log', regularization='l2', lamb=0.1)
    nn.split_data(frac=0.5, shuffle=True)
    nn_l2.split_data(frac=0.5, shuffle=True)
    nn_l2.split_data(frac=0.5, shuffle=True)
    print("Normal")
    nn.TrainNN(epochs = 300, eta0 = 0.01, n_print=10)
    print("L1")
    nn_l1.TrainNN(epochs = 300, eta0 = 0.01, n_print=10)
    print("L2")
    nn_l2.TrainNN(epochs = 300, eta0 = 0.01, n_print=10)



check_regularization()

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

# # # print("Critical error: %g,  Critical accuracy: %g" %(critError, critAcc))




# # # ### LOG-REG CASE
# # # # nn = NeuralNet(X,Y, nodes = [X.shape[1], 2], activations = [None], cost_func='log')
# # # # nn.split_data(frac=0.5, shuffle=True)
# # # # nn.feed_forward(nn.xTrain)
# # # # nn.backpropagation()
# # # # nn.TrainNN(epochs = 100, eta = 0.001, n_print=5)
