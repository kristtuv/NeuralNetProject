import numpy as np
from sklearn import linear_model

np.random.seed() # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit


##### prepare training and test data sets
import pickle,os
from sklearn.model_selection import train_test_split

###### define ML parameters
num_classes=2
train_to_test_ratio=0.5 # training samples

# path to data directory
#path_to_data=os.path.expanduser('~')+'/Dropbox/MachineLearningReview/Datasets/isingMC/'

# load data
file_name = "IsingData/Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "IsingData/Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

# full data set
X=np.concatenate((X_critical,X))
Y=np.concatenate((Y_critical,Y))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')


X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)


# Shuffle training data
randomize = np.arange(X_train.shape[0])
np.random.shuffle(randomize)
X_train = X_train[randomize]
Y_train = Y_train[randomize]

J = np.random.uniform(-0.5, 0.5, (X_train.shape[1],1))

m = X_train.shape[0]
batchSize = int(X_train.shape[0]/m)
print("Batch size: ", batchSize)

epochs = 100
eta = 0.00001
lamb = 100


"""
logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=lamb, max_iter=200,
                                           shuffle=True, random_state=1, learning_rate='optimal')

# fit training data
logreg_SGD.fit(X_train,Y_train)

train_accuracy_SGD=logreg_SGD.score(X_train,Y_train)
test_accuracy_SGD=logreg_SGD.score(X_test,Y_test)
print(train_accuracy_SGD)
print(test_accuracy_SGD)
print(logreg_SGD.intercept_)
"""

#Stochastic Gradient Descent (SGD)
for epoch in range(epochs + 1):

    for i in range(m):

        rand_idx = np.random.randint(m)

        xBatch = X_train[rand_idx*batchSize : (rand_idx+1)*batchSize]
        yBatch = Y_train[rand_idx*batchSize : (rand_idx+1)*batchSize]

        y = xBatch @ J
        p = np.exp(y)/(1 + np.exp(y))
        dJ = -(xBatch.T @ (yBatch - p) - 2*lamb*J)/xBatch.shape[0]
        J -= eta*dJ

    if epoch % 10 == 0 or epoch == 0:
        #cost = -np.sum(yBatch * (xBatch @ J) - np.log(1 + np.exp(xBatch @ J)))/xBatch.shape[0]
        #accuracy = np.sum((p > 0.5) == yBatch)/xBatch.shape[0]
        logit_train = X_train @ J
        train_cost = (-np.sum((Y_train * logit_train) - np.log(1 + np.exp(logit_train))) + lamb*np.sum(J**2))/X_train.shape[0]
        p_train = np.exp(logit_train)/(1 + np.exp(logit_train))
        train_accuracy = np.sum((p_train > 0.5) == Y_train)/X_train.shape[0]

        logit_test = X_test @ J
        test_cost = (-np.sum((Y_test * logit_test) - np.log(1 + np.exp(logit_test))) + lamb*np.sum(J**2))/X_test.shape[0]
        p_test = np.exp(logit_test)/(1 + np.exp(logit_test))
        test_accuracy = np.sum((p_test > 0.5) == Y_test)/X_test.shape[0]

        print("Epoch: ", epoch)
        print("  Cost  | Training: %f, Test: %f" %(train_cost, test_cost))
        print("Accuracy| Training: %f, Test: %f" %(train_accuracy, test_accuracy))
        print("-"*50)
