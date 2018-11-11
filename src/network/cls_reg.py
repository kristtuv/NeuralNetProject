import numpy as np
import scipy.linalg as scl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from types import  MethodType
from tqdm import tqdm
import sys
import matplotlib.pylab as plt


def check_types(*args):
    """
    Decorator for checking types of function arguments
    Used during testing
    """
    def decorator(func):
        def wrapper(*argswrapper):
            argswrappercopy = (argswrapper[1:])
            for a, b in zip(args, argswrappercopy):
                if a is not type(b) and type(b) is not type(None) :
                    raise TypeError('See documentation for argument types')
            return func(*argswrapper)
        return wrapper
    return decorator



class LinReg:
    # @check_types(np.ndarray, np.ndarray, np.ndarray, int)
    def __init__(self, xData, yData):
        """
        :param x: 1darray of x values
        :param y: 1darray of y values
        :param z: The values we are trying to fit
        :param deg: The degree of polynomial we try to fit the data
        :type x: ndarray
        :type y: ndarray
        :type z: ndarray
        :type deg: int
        """

        self.xData = xData
        self.yData = yData
        self.N = xData.shape[0]

        self.split_data(folds = 10, frac = 0.3)


    def split_data(self, folds = None, frac = None, shuffle = False):
        """
        Splits the data into training and test. Give either frac or folds

        param: folds: Number of folds
        param: frac: Fraction of data to be test data
        param: shuffle: If True: shuffles the design matrix
        type: folds: int
        type: frac: float
        type: shuffle: Bool
        return: None
        """

        if folds == None and frac == None:
            print("Error: No split info received, give either no. folds or fraction.")
            sys.exit(0)

        xData = self.xData
        yData = self.yData

        if shuffle:
            randomize = np.arange(xData.shape[0])
            np.random.shuffle(randomize)
            xData = xData[randomize]
            yData = yData[randomize]

        if folds != None:
            xFolds = np.array_split(xData, folds, axis = 0)
            yFolds = np.array_split(yData, folds, axis = 0)

            self.xFolds = xFolds
            self.yFolds = yFolds

        if frac != None:
            nTest = int(np.floor(frac*xData.shape[0]))
            xTrain = xData[:-nTest]
            xTest = xData[-nTest:]

            yTrain = yData[:-nTest]
            yTest = yData[-nTest:]

            self.xTrain = xTrain ; self.xTest = xTest
            self.yTrain = yTrain ; self.yTest = yTest
            self.nTrain = xTrain.shape[0] ; self.nTest = xTest.shape[0]


    # @check_types(np.ndarray, np.ndarray)
    def ols(self, xData = None, yData = None):
        """
        Performes a Ordinary least squares linear fit

        :param xData: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :type xData: array
        :type z: array
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if xData is None: xData = self.xData
        if yData is None: yData = self.yData

        #beta = scl.inv(xData.T @ xData) @ xData.T @ z
        beta = np.linalg.pinv(xData) @ yData
        """
        ypredict = xData @ beta
        vary = 1.0/(xData.shape[0])*np.sum((yData - ypredict)**2)
        var = np.linalg.pinv(xData.T @ xData)*vary
        self.var_ols = var
        self.conf_ols = 1.96*np.sqrt(np.diag(var))
        """
        return beta

    # @check_types(np.ndarray, np.ndarray)
    def ridge(self, xData = None, yData = None):
        """
        Performes a Ridge regression linear fit

        :param xData: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :type xData: array
        :type z: array
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if xData is None: xData = self.xData
        if yData is None: yData = self.yData

        I = np.identity(xData.shape[1])
        xData_inv = scl.inv(xData.T @ xData + self.lamb*I)
        beta = xData_inv @ xData.T @ yData
        """
        ypredict = xData @ beta
        vary = 1.0/(xData.shape[0])*np.sum((yData - ypredict)**2)

        var = xData_inv @ xData.T @ xData @ xData_inv.T * vary
        self.var_ridge = var
        self.conf_ridge = 1.96*np.sqrt(np.diag(var))
        """
        return beta


    # @check_types(np.ndarray, np.ndarray)
    def lasso(self, xData = None, yData = None):
        """
        Performes a Lasso regression linear fit

        :param xData: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :type xData: array
        :type z: array
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if xData is None: xData = self.xData
        if yData is None: yData = self.yData

        lass = Lasso([float(self.lamb)], fit_intercept=False)#, max_iter = 5000)
        lass.fit(xData, yData)

        beta = (lass.coef_).reshape(xData.shape[1], 1)

        return beta

    # @check_types(np.ndarray, np.ndarray)
    def MSE(self, yData, ypred):
        """
        Finds the mean squared error of the real data and predicted values

        :param z: real data
        :param zpred: predicted data
        :type z: array
        :type zpred: array
        :return: The mean squared error
        :rtype: float
        """

        return 1.0/yData.shape[0]*np.sum((yData - ypred)**2)


    # @check_types(np.ndarray, np.ndarray)
    def R2(self, yData, ypred):
        """
        Finds the R2 error of the real data and predicted values

        :param z: real data
        :param zpred: predicted data
        :type z: array
        :type zpred: array
        :return: The mean squared error
        :rtype: float
        """


        ymean = np.average(yData)

        return 1 - np.sum((yData - ypred)**2)/np.sum((yData - ymean)**2)

    #@check_types(int, MethodType)
    def bootstrap(self, nBoots, regressionmethod):
        """
        Bootstraps the data defined in the instance
        and calculates the bias, variance, and average
        mse training error and mse test error

        param: nBoots: Number of boostrap samples
        param: regressionmethod: ols, ridge or lasso
        type: nBoots: int
        type: regressionmethod: MethodType
        return: Bias, Variance, Training Error, Test Error
        rtype: float, float, float, float
        """
        nTrain = self.xTrain.shape[0]
        nTest = self.xTest.shape[0]

        ypreds = np.zeros((nBoots, nTest))
        train_errors = np.zeros(nBoots)
        test_errors = np.zeros(nBoots)
        train_r2 = np.zeros(nBoots)
        test_r2 = np.zeros(nBoots)

        for i in tqdm(range(nBoots)):
            idx = np.random.choice(nTrain, nTrain)
            xboot = self.xTrain[idx]
            yboot = self.yTrain[idx]

            beta = regressionmethod(xboot, yboot)
            ypred_train = xboot @ beta
            ypred_test = self.xTest @ beta

            ypreds[i] = ypred_test.flatten()
            train_errors[i] = self.MSE(yboot, ypred_train)
            test_errors[i] = self.MSE(self.yTest, ypred_test)
            train_r2[i] = self.R2(yboot, ypred_train)
            test_r2[i] = self.R2(self.yTest, ypred_test)

        train_error = np.average(train_errors)
        test_error = np.average(test_errors)
        train_r2 = np.average(train_r2)
        test_r2 = np.average(test_r2)

        y_avg = np.average(ypreds, axis = 0).reshape(-1, 1)
        variance = np.average(np.var(ypreds, axis = 0))
        bias = np.average((self.yTest - y_avg)**2)

        return bias, variance, train_error, test_error, train_r2, test_r2

    # @check_types(int, MethodType)
    def kfold(self, nfolds, regressionmethod):
        """
        Split data into folds and run k-fold algorithm
        and calculate average r2 and mse error for
        for training and test

        param: nfolds: number of folds
        type: nfolds: int
        param: regressionmethod: ols, ridge or lasso
        type: regressionmethod: MethodType
        return: mse_train, mse_test, r2_train, r2_test
        rtype: float, float, float, float
        """
        if nfolds != 10:
            self.split_data(folds = nfolds)

        xFolds = self.xFolds
        yFolds = self.yFolds

        mse_train = 0; mse_test = 0
        r2_train = 0; r2_test = 0

        for i in range(nfolds):

            xTrain = xFolds.copy()
            xTest = xTrain.pop(i)
            xTrain = np.concatenate(xTrain)

            yTrain = yFolds.copy()
            yTest = yTrain.pop(i)
            yTrain = np.concatenate(yTrain)

            beta = regressionmethod(xTrain, yTrain)
            ypred_train = xTrain @ beta
            ypred_test = xTest @ beta

            mse_train += self.MSE(yTrain, ypred_train)
            mse_test += self.MSE(yTest, ypred_test)
            r2_train += self.R2(yTrain, ypred_train)
            r2_test += self.R2(yTest, ypred_test)

        mse_train /= nfolds; mse_test /= nfolds
        r2_train /= nfolds; r2_test /= nfolds

        return mse_train, mse_test, r2_train, r2_test
