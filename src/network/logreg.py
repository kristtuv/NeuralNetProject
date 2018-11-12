import numpy as np
from sklearn import linear_model
import matplotlib.pylab as plt
import pickle,os
from sklearn.model_selection import train_test_split


class LogReg_Ising():

    def __init__(self):

        self.fetch_data()

        # Initialize parameters
        self.J = np.random.uniform(-0.5, 0.5, (self.X_train.shape[1],1))

    def fetch_data(self):
        """
        Fetchin the 2 dimensional ising data
        and spliting it into ordered, disordered and critical
        data
        """

        train_to_test_ratio=0.5 # training samples

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
        self.X_critical = np.c_[np.ones(X_critical.shape[0]), X_critical]
        self.Y_critical = Y_critical

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


        self.X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        self.X_test = np.c_[np.ones(X_test.shape[0]), X_test]

        self.Y_train = Y_train.reshape(-1, 1)
        self.Y_test = Y_test.reshape(-1, 1)



    def optimize(
        self,
        method = 'SGD',
        m = 100,
        epochs = 100,
        eta = 'schedule',
        regularization=None,
        lamb = 0.0):

        """
        param: method: What type of gradient descent optimiser
        param: m: Iterations within one epoch
        param: epochs: Number of full iterations through data set
        param: eta: Learing rate
        param: regularization: Type of regularization
        param: lamb: Regularization strength
        type: method: string
        type: m: int
        type: epochs: int
        type: eta: string
        type: regularization: string
        type: lamb: float
        """

        X_train = self.X_train ; X_test = self.X_test
        Y_train = self.Y_train ; Y_test = self.Y_test
        J = self.J

        batchSize = int(X_train.shape[0]/m)
        self.batchSize = batchSize
        print("Batch size: ", batchSize)

        if eta == 'schedule':
            t0 = 5 ; t1 = 50
            learning_schedule = lambda t : t0/(t + t1)
        else:
            learning_schedule = lambda t : eta

        if regularization == 'l2':
            reg_cost = lambda J: lamb*np.sum(J**2)
            reg_grad = lambda J: 2*lamb*J
        elif regularization == 'l1':
            reg_cost = lambda J: lamb*np.sum(np.abs(J))
            reg_grad = lambda J: lamb*np.sign(J)
        elif regularization == None:
            reg_cost = lambda J: 0
            reg_grad = lambda J: 0

        #Stochastic Gradient Descent (SGD)
        for epoch in range(epochs + 1):

            # Shuffle training data
            randomize = np.arange(X_train.shape[0])
            np.random.shuffle(randomize)
            X_train = X_train[randomize]
            Y_train = Y_train[randomize]

            for i in range(m):

                rand_idx = np.random.randint(m)

                xBatch = X_train[rand_idx*batchSize : (rand_idx+1)*batchSize]
                yBatch = Y_train[rand_idx*batchSize : (rand_idx+1)*batchSize]

                y = xBatch @ J
                #p = np.exp(y)/(1 + np.exp(y))
                p = 1.0/(1 + np.exp(-y))
                eta = learning_schedule(epoch*m+i)
                dJ = -(xBatch.T @ (yBatch - p))/xBatch.shape[0] + reg_grad(J)
                J -= eta*dJ

            if epoch % 10 == 0 or epoch == 0:

                logit_train = X_train @ J
                self.train_cost = (-np.sum((Y_train * logit_train) - np.log(1 + np.exp(logit_train))))/X_train.shape[0] + reg_cost(J)
                p_train = 1/(1 + np.exp(-logit_train))
                self.train_accuracy = np.sum((p_train > 0.5) == Y_train)/X_train.shape[0]

                logit_test = X_test @ J
                self.test_cost = (-np.sum((Y_test * logit_test) - np.log(1 + np.exp(logit_test))))/X_test.shape[0] + reg_cost(J)
                p_test = 1/(1 + np.exp(-logit_test))
                self.test_accuracy = np.sum((p_test > 0.5) == Y_test)/X_test.shape[0]

                print("Epoch: ", epoch)
                print("  Cost  | Training: %f, Test: %f" %(self.train_cost, self.test_cost))
                print("Accuracy| Training: %f, Test: %f" %(self.train_accuracy, self.test_accuracy))
                print("-"*50)


if __name__ == '__main__':

    logreg = LogReg_Ising()
    logreg.optimize(m = 1000, regularization='l1', lamb = 100)
