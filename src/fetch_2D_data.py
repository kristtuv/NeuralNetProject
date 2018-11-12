import numpy as np
import pickle,os
from sklearn.model_selection import train_test_split
def fetch_data(ordered_stop=70000, disordered_start=100000, crit=True):
    """
    Fetchin the 2 dimensional ising data
    and spliting it into ordered, disordered and critical
    data. Returns ordered and unordered states as a single matrix.

    param: ordered_stop: stop value for the ordered samples
    param: disordered_start: start value for the disordered samples
    param: crit: set to True of critical values should be returned
    type: ordered_stop: int
    type: disordered_start: int
    type: crit: bool
    return: Matrices of training samples X, Y, X_critical and Y_critical
    or X and Y
    """

    # load data
    file_name = "IsingData/Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
    data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
    data=data.astype('int')
    data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

    file_name = "IsingData/Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    labels = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

    # divide data into ordered, critical and disordered
    X_ordered=data[:ordered_stop,:]
    Y_ordered=labels[:ordered_stop]
    X_disordered=data[disordered_start:,:]
    Y_disordered=labels[disordered_start:]


    # define training and test data sets
    X=np.concatenate((X_ordered,X_disordered))
    Y=np.concatenate((Y_ordered,Y_disordered))
    if crit:
        X_critical=data[ordered_stop:disordered_start,:]
        Y_critical=labels[ordered_stop:disordered_start]
        del data, labels
        return X, Y, X_critical, Y_critical

    else:
        del data,labels
        return X, Y


if __name__ == '__main__':
    fetch_data()
