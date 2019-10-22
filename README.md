# Ising Model Analysis using linear regregreesion and neural networks   
Project doing linear regression, logistic regression and binary classification using a simple neural network on data from the 1D and 2D Ising model    
## Build with    
scikit-image    0.13.1      
scikit-learn    0.19.2      
python          3.6.2      
numpy           1.13.3      
matplotlib      2.0.2       
      
    
## Structure of the repo    
### IsingData    
Containg the data for two dimensional Ising Model    
    
### Metha    
Notebooks from Metha et al    
### csv    
csv files for neural network runs    
### plots    
plots for report    
### results    
Results from linear regression and logistic regression simulations  
### src    
**NN_fulldata.py:** Program for testing neural network architecture on full dataset from the two dimensional Ising model    
**NN_partialdata.py:** Program for testing neural network architecture on partial dataset from the two dimensional Ising model   
**NN_lin.py:** Program for testing neural network architecture on dataset from the one dimensional Ising model   
**NN_log.py:** Program for testing neural network architecture onpartial dataset    
**linreg_ising.py:** Testing linear regression models on the one dimensional Ising model
**logreg_main.py:** Testing logistic regression on the one dimensional Ising model    
**fetch_2D_data.py:** Handeling the data for 2D ising model       
**misc:** Contians programs used to generate example plots or testing concepts    
**network:** Package    
* **NN.py:** Class containing neural network methods    
* **cls_reg.py:** Class containing linear regression methods ols, ridge and lasso; resampling k-fold and bootstrap; statistics MSE and R2    
* **logreg.py:** Class for doing logistic regression    
* **plotparams.py:** Parameters for ploting    
    
    
## Authors    
    
* **Tommy Myrvik** - (https://github.com/myrvixen)    
* **Kristian Tuv** - (https://github.com/kristtuv)    
