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
    
    def sigmoid_derivative(dA, activation_cache):
        Z = activation_cache
        s = 1/ (1+np.exp(-z))
        dZ = dA * s * (1-s)
        
        return dZ
    
    @staticmethod 
    def relu(z):
        return np.maximum(0,z)
    
    def relu_derivative(dA, activation_cache):
        dZ = np.array(dA, copy= True)
        dZ[activation_cache <= 0] = 0
    
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
            A = self.sigmoid(Z)
            
        elif activation.lower() == "relu":
            A = self.relu(Z)
            
        activation_cache = Z
        
        cache = (linear_cache, activation_cache)
        
        return A, cache
    
    def forward_propagation(self, parameters):
        
        caches = []
        A = self.X
        num_layers = len(parameters) // 2
        
        for layer_num in range(1,num_layers):
            A_prev = A
            A, cache = self.forward(A_prev, parameters["W"+str(layer_num)], parameters["b"+str(layer_num)],"relu")
            caches.append(cache)
            
        AL, cache = self.forward(A, parameters["W"+str(num_layers)], parameters["b"+str(num_layers)],"sigmoid")
        caches.append(cache)
        
        return AL, caches
        
    def calculate_cost(self, AL):
        
        num_samples = self.y.shape[1]
        
        cost = -(1/num_samples) * (np.sum((np.dot(self.Y,np.log(AL.T)))+ (np.dot(1-self.Y,np.log(1-AL.T)))))
        
        cost.np.squeeze(cost)
        
        return cost
        
    def backward(self, dA, cache, activation):
        
        linear_cache, activation_cache = cache
        A_prev, W, b = linear_cache
        num_samples = self.y.shape[1]
        
        if activation == "relu":
            dZ = self.relu_derivative(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_derivative(dA, activation_cache)
            
        dW = (1/num_samples) * np.dot(dZ,A_prev.T)
        db = (1/num_samples) * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)      
        
        return dA_prev, dW, db      
        
    def backpropagation(self, AL, caches):
        
        gradients = {}
        num_layers = len(self.num_nodes)
        num_samples = self.y.shape[1]
        self.y = self.y.reshape(AL.shape)
        
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        current_cache = caches[num_layers-1]
        
        dA_prev_temp, dW_temp, db_temp = self.backward(dAL, current_cache,"sigmoid")
        
        for layer in reversed(range(num_layers-1)):
            current_cache = caches[layer]
            dA_prev_temp, dW_temp, db_temp = self.backward(dA_prev_temp, current_cache, "relu")
            gradients["dA"+str(layer)] = dA_prev_temp
            gradients["dW"+str(layer+1)] = dW_temp
            gradients["db"+str(layer+1)] = db_temp
            
        return gradients

    def update_parameters(self, parameters, gradients):
        updated_parameters = parameters.copy()
        num_layers = len(self.num_nodes)
        
        for layer in range(num_layers):
            updated_parameters["W"+str(layer+1)] = updated_parameters["W"+str(layer+1)] - np.dot(self.lr,gradients["dW"+str(layer+1)])
            updated_parameters["b"+str(layer+1)] = updated_parameters["b"+str(layer+1)] - np.dot(self.lr,gradients["db"+str(layer+1)])
            
        return updated_parameters
    
           
    
X = np.random.rand(10, 100)
y = np.random.rand(5, 100)
num_nodes = [50, 30, 20]
model = DeepNeuralNetwork(X,y,num_nodes)

print(model.initialise_weights_bias().keys())