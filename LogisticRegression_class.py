import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from prepare_image_data import prepare_image_data

class LogisticRegression:
    
    def __init__(self, X, y, num_iterations = 100, lr = 0.001):
        self.X =  X
        self.Y =  y
        self.m = X.shape[1]
        self.w, self.b = self.initialise_weights_bias()
        self.num_iterations = num_iterations
        self.lr = lr
        
        params, gradients, cost = self.update_weights_bias()
        self.Y_pred_train = self.predict(self.X, self.Y)
        
    @staticmethod    
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def initialise_weights_bias(self):
        input_shape = self.X.shape[0]
        w = np.zeros((input_shape,1))
        b = float(0)
        
        return w,b 
    
    def forward(self):
        
        Z = np.dot(self.w.T,self.X) + self.b
        A = self.sigmoid(Z)
        cost = -(1/self.m) * (np.sum((np.dot(self.Y,np.log(A.T))) + (np.dot(1-self.Y,np.log(1-A.T)))))
        
        return A, cost
        
    def backward(self, A, cost):
        
        dw = (1/self.m) * np.dot(self.X,np.transpose(A-self.Y))
        db = (1/self.m) * np.sum(A-self.Y)
        
        cost = np.squeeze(np.array(cost))
        
        gradient = {"dw":dw, "db":db}
        
        return gradient, cost
        
    def update_weights_bias(self):
        
        for idx in range(self.num_iterations):
            
            A, cost = self.forward()
            gradient, cost = self.backward(A, cost)
            
            dw = gradient["dw"]
            db = gradient["db"]
            
            self.w -= self.lr*dw
            self.b -= self.lr*db
            
            if idx % 50 == 0:
                print(f"The cost after {idx} iterations is {cost}")
        
        params = {"w":self.w, "b":self.b}
        gradients = {"dw":dw,"db":db}
        
        return params, gradients, cost       
        
    def predict(self,X,Y):
        
        test_size = X.shape[1]
        Y_pred = np.zeros((1,test_size))
        self.w = self.w.reshape(X.shape[0],1)
        
        A = self.sigmoid((np.dot(self.w.T,X))+ self.b)
        
        for i in range(test_size):
            
            if A[0,i] > 0.5:
                Y_pred[0,i] = 1
            else:
                Y_pred[0,i] = 0
                
        self.print_accuracy(Y_pred, Y)
                
        return Y_pred
        
    def __str__(self):
        return "Training accuracy: {}%" .format(100 - np.mean(np.abs(self.Y_pred_train - self.Y)) * 100)
    
    def print_accuracy(self,Y_pred,Y):
        print("Accuracy: {}%" .format(100 - np.mean(np.abs(Y_pred - Y)) * 100))