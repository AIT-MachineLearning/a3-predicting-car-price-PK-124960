import streamlit as st
import pickle
import numpy as np
from sklearn.model_selection import KFold

class MultinomialLogisticRegression:
    def __init__(self, regularization, learning_rate=0.01, epochs=1000, num_classes=4):
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes
        self.intercept_ = None

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # For numerical stability
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def cross_entropy_loss(self, A, y):
        m = y.shape[0]
        log_likelihood = -np.log(A[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def gradient(self, X, A, y):
        m = X.shape[0]
        A[range(m), y] -= 1  # Subtract 1 from the correct class probabilities
        grad_theta = np.dot(X.T, A) / m
        grad_intercept = np.sum(A, axis=0) / m  # Gradient for intercept term
        return grad_theta, grad_intercept

    def fit(self, X, y):
        m, n_features = X.shape
        
        # Initialize weights (theta) and intercept
        self.theta = np.random.randn(n_features, self.num_classes) * 0.01
        self.intercept_ = np.zeros(self.num_classes)  # Initialize intercepts

        for epoch in range(self.epochs):
            # Forward pass: calculate predictions
            Z = np.dot(X, self.theta) + self.intercept_  # Include intercept
            A = self.softmax(Z)

            # Compute the loss with regularization
            loss = self.cross_entropy_loss(A, y)
            loss += self.regularization(self.theta)  # Add penalty to the loss

            # Backward pass: compute gradients
            grad_theta, grad_intercept = self.gradient(X, A, y)
            grad_theta += self.regularization.derivation(self.theta)  # Add penalty to the gradient

            # Update weights and intercept
            self.theta -= self.learning_rate * grad_theta
            self.intercept_ -= self.learning_rate * grad_intercept

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        Z = np.dot(X, self.theta) + self.intercept_  # Include intercept in prediction
        A = self.softmax(Z)
        return np.argmax(A, axis=1)  # Return class with the highest probability

class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):  # __call__ allows us to call the class as a method
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class LinearPenalty:
    def __init__(self, l):
        self.l = l  

    def __call__(self, theta): 
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)
    
    
class Ridge(MultinomialLogisticRegression):
    def __init__(self, learning_rate, l, epochs=1000):
        regularization = RidgePenalty(l)
        super().__init__(regularization, learning_rate, epochs)


class Linear(MultinomialLogisticRegression):
    def __init__(self, learning_rate, l, epochs=1000):
        regularization = LinearPenalty(l)
        super().__init__(regularization, learning_rate, epochs)

# if(__name__ == "__main__"):
#     import pickle
#     from LinearClassA3 import *
#     import os
#     print(os.getcwd())
#     base_path:str = os.getcwd()
#     full_path = os.path.join(base_path, "trained_model_v3.sav")
#     print(full_path)