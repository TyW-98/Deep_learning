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
    
    @staticmethod 
    def relu(z):
        return np.maximum(0,z)
    
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
    
    def forward(self, A, W, b, activation):
        
        Z = np.dot(W,A) + b
        linear_cache = (A, W, b)
        
        if activation.lower() == "sigmoid":
            A = sigmoid(Z)
            
        elif activation.lower() == "relu":
            A = relu(Z)
            
        activation_cache = Z
        
        cache = (linear_cache, activation_cache)
        
        return A, cache
    
    def forward_propagation(self, parameters):
        
        caches = []
        A = self.X
        num_layers = len(parameters) // 2
        
        for layer_num in range(1,num_layers):
            A_prev = A
            A, cache = forward(A_prev, parameters["W"+str(layer_num)], parameters["b"+str(layer_num)],"relu")
            caches.append(cache)
            
        AL, cache = forward(A, parameters["W"+str(num_layers)], parameters["b"+str(num_layers),"sigmoid")
        caches.append(cache)
        
        return AL, caches
        
    def calculate_cost(self, AL):
        
        num_samples = self.y.shape[1]
        
        cost = -(1/num_samples) * (np.sum((np.dot(self.Y,np.log(AL.T)))+ (np.dot(1-self.Y,np.log(1-AL.T)))))
        
        cost.np.squeeze(cost)
        
        return cost
        
    
X = np.random.rand(10, 100)
y = np.random.rand(5, 100)
num_nodes = [50, 30, 20]
model = DeepNeuralNetwork(X,y,num_nodes)

print(model.initialise_weights_bias().keys())