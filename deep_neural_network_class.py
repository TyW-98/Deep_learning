import numpy as np
import os
import matplotlib.pyplot as plt
from prepare_image_data import prepare_image_data

class DeepNeuralNetwork:
    
    def __init__(self,X,y,num_nodes_each_layer,num_iterations = 100, lr=0.001):
        self.X = X
        self.y = y
        self.num_nodes = num_nodes_each_layer
        self.num_iterations = num_iterations
        self.lr = lr
        
    @staticmethod    
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def initialise_weights_bias(self):
        input_shape = self.X.shape[0]
        parameters = {}
        
        for num_layer in range(len(self.num_nodes)):
            
            if num_layer == len(self.num_nodes) - 1:
                parameters["W"+str(len(self.num_nodes))] = np.random.randn(self.y.shape[0],self.num_nodes[-1]) * 0.01
                parameters["b"+str(len(self.num_nodes))] = np.zeros((self.y.shape[0],1))
            elif num_layer == 0:
                parameters["W1"] = np.random.randn(self.num_nodes[1],input_shape) * 0.01
                parameters["b1"] = np.zeros((self.num_nodes[1],1))
            else:
                parameters["W"+str(num_layer+1)] = np.random.randn(self.num_nodes[num_layer],self.num_nodes[num_layer-1]) * 0.01
                parameters["b"+str(num_layer+1)] = np.zeros((self.num_nodes[num_layer],1))
                
        return parameters
    

X = np.random.rand(10, 100)
y = np.random.rand(5, 100)
num_nodes = [50, 30, 20]
model = DeepNeuralNetwork(X,y,num_nodes)

print(model.initialise_weights_bias().keys())