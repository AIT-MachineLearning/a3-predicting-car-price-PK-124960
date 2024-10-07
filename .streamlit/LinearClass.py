import streamlit as st
import pickle
import numpy as np
from sklearn.model_selection import KFold

class LinearRegression(object):
        
    def __init__(self, regularization, lr=0.001, method='mini', initialization='zero', step="mom", polynomial = False, 
                 num_epochs=500, batch_size=50, cv=KFold(n_splits=3), momentum=0.85):
        
        self.lr             = lr
        self.num_epochs     = num_epochs
        self.batch_size     = batch_size
        self.method         = method
        self.cv             = cv
        self.regularization = regularization
        self.initialization = initialization
        self.step           = step
        self.prev_step      = 0
        self.momentum       = momentum
        self.polynomial     = polynomial
    
    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    def fit(self, X_train, y_train):
            
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty
        
        # self.columns = X_train.columns
        
        if self.polynomial == True:
            X_train._transform_features(X_train)
        else:
            # X_train = X_train.to_numpy()
            pass

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx] #KeyError
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            #Add Initialization Choices
            if self.initialization == 'xavier':
                #Xavier Initialization
                m = X_cross_train.shape[1]
                #calculate range of weights
                lower, upper = -1.0/np.sqrt(m), 1.0/np.sqrt(m)  
                #randomly pick weights 
                xavier_weight = lower + np.random.rand(m) * (upper-lower)
                #Create theta with size of features (10, ) using xavier init
                self.theta = xavier_weight
                # print(f"Xavier Weight: {xavier_weight}")
            else: #Zeros Initialization
                self.theta = np.zeros(X_cross_train.shape[1])
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            #with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                params = {"method": self.method, "xavier":self.initialization, "step":self.step , "lr": self.lr, "reg": type(self).__name__}
                #mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'stochastic':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini-batch':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    # mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    # mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
            
                    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        #Add Momentum Step
        if self.step == "momentum":
            step = self.lr * grad
            #Calculate step : new-theta = old-theta + momentum * previous-step
            self.theta = self.theta - step + self.momentum * self.prev_step
            #Assign step as prev_step
            self.prev_step = step
        else:
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
    
    def predict(self, X):
        #Add check isPolynomial
        if self.polynomial == True:
            X = self._transform_features(X)
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def mse(self, ypred, ytrue):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    # Add r2_score function
    def r2_score(self, ypred, ytrue):
        residuals = ytrue - ypred
        total_sum_of_squares = np.sum((ytrue - np.mean(ytrue)) ** 2)
        residual_sum_of_squares = np.sum(residuals ** 2)
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2
        
#Add new class for normal LinearRegression
class LinearPenalty:

    def __init__(self, l):
        self.l = l  

    def __call__(self, theta): 
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)

class Linear(LinearRegression):

    def __init__(self, method, lr, l, initialization, step, polynomial):
        self.regularization = LinearPenalty(l)
        super().__init__(self.regularization, lr, method, initialization, step, polynomial)

    def get_coefficients(self):
        # Access the method from the parent class
        return self._coef()


if(__name__ == "__main__"):
    import pickle
    from LinearClass import *
    import os
    print(os.getcwd())
    base_path:str = os.getcwd()
    full_path = os.path.join(base_path, "trained_model_v2.1.sav")
    print(full_path)
    # filename = '\.streamlit\trained_model_v2.1.sav'
    # #filename = 'D:/AT82.03_ML/A1_Predicting_Car_Price/a1-predicting-car-prices-PK-124960/.streamlit/st124960_car_predict.model'
    # loaded_model = pickle.load(open(filename, 'rb'))