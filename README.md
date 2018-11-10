# Linear Regression analysis       
Project doing linear regression, logistic regression and binary classification using a simple neural network on data from the 1D and 2D Ising model    
## Build with    
scikit-image    0.13.1      
scikit-learn    0.19.2      
pytest          3.2.1       
python          3.6.2      
numpy           1.13.3      
matplotlib      2.0.2       
tqdm            4.23.4          
      
    
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
Results from hva?    
    
### src    
**NN_fulldata.py:** Program for testing neural network architecture on full dataset    
**NN_partialdata.py:** Program for testing neural network architecture on partial dataset    
**linreg_ising.py:** ?    
**logreg_main.py:** ?    
**main.py:** ?    
**scikit_temp.py:** ?    
**misc:** Contians programs used to generate example plots or testing concepts    
**network:** package    
* **NN.py:** Class containing neural network methods    
* **cls_reg.py:** Class containing linear regression methods ols, ridge and lasso; resampling k-fold and bootstrap; statistics MSE and R2    
* **logreg.py:** Class for doing logistic regression    
* **fetch_2D_data.py:** Handeling the data for 2D ising model    
* **plotparams.py:** Parameters for ploting    
    
    
> ## Running the tests    
> Run unit_test.py with pytest -v      
> pytest unitest.py -v    
    
    
    
## Authors    
    
* **Tommy Myrvik** - (https://github.com/tommymy)    
* **Kristian Tuv** - (https://github.com/kristtuv)    
