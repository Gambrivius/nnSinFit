#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
import time

amplitude = 2
offset = 0
frequency = 5
samples = 1000

batchsize = 15
learning_rate = 0.0001

x0 = np.linspace(-1, 1, samples)

def true_y (x):
    return np.sin(x*frequency)*amplitude+offset

y0 = true_y (x0)
plt.plot(x0, y0)
plt.xlim ([-1,1])
plt.ylim ([-amplitude*1.1,amplitude*1.1])
plt.show()


# In[ ]:





# In[12]:



class Layer_Dense:
    
    def __init__ (self, n_inputs, n_neurons):
        self.weights =  0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward (self, inputs):
        self.inputs = inputs
        self.output = np.dot (inputs, self.weights) + self.biases


    def backward (self, dvalues):
        self.dweights = np.dot (self.inputs.T, dvalues)
        self.dbiases = np.sum (dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot (dvalues, self.weights.T)
        
    def update_params (self):
        self.weights += -learning_rate * self.dweights
        self.biases += -learning_rate * self.dbiases
            

class Activation_ReLU:
    def forward (self, inputs):
        self.inputs = inputs
        self.output = np.maximum (0, inputs)
    def backward (self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class LossMSE:
    def forward (self, inputs, y_true):
        # calculated the mse for all samples
        self.inputs = inputs
        self.loss = np.mean((inputs - y_true)**2)
        self.output = inputs
        
    def backward (self, dvalues, ytrue):

        self.dinputs = 2*(dvalues - ytrue)

        
        
# model structure
dense1 = Layer_Dense (1, 25)
activation1 = Activation_ReLU()
dense2 = Layer_Dense (25, 150)
activation2 = Activation_ReLU()
dense3 = Layer_Dense (150, 1)

loss_activation = LossMSE()

def forward_all (inputs):
    dense1.forward (inputs)
    activation1.forward (dense1.output)
    dense2.forward (activation1.output)
    activation2.forward (dense2.output)
    dense3.forward (activation2.output)
    loss_activation.forward (dense3.output, true_y(inputs))

def backward_all (inputs):
    loss_activation.backward (loss_activation.output,  true_y(inputs))
    dense3.backward (loss_activation.dinputs)
    activation2.backward (dense3.dinputs)
    dense2.backward (activation2.dinputs)
    activation1.backward (dense2.dinputs)
    dense1.backward (activation1.dinputs)
                            

s = np.sort(np.random.random((15,1))*2.5-1.2,axis = 0)
forward_all ( s)
def visualize():
    y1 = dense3.output.flatten()
    plt.plot(x0, y0)
    x1 = s.flatten()
    plt.plot(x1, y1)
    
    plt.show()

def visualize2():
    x1 = np.linspace(-1, 1, 100)
    y0 = true_y (x1)
    forward_all (x1.reshape(100,1))
    y1 = dense3.output.flatten()
    plt.plot(x1, y0)
    plt.plot(x1, y1)
    plt.xlim ([-1,1])
    plt.ylim ([-amplitude*1.1,amplitude*1.1])
    error = np.mean((y1-y0)**2)
    print ("MSE: ", error)
    plt.show()
    
    return error
    
visualize2()


# In[13]:



# stochastic gradient descent loop
mse = visualize2()
n = 0

while (mse > 0.0003):
    n += 1
    
    # s = a set of x samples between -2.5 and 2.5 of size batchsize
    
    s = np.random.random((batchsize,1))*2.5-1.25
    
    forward_all (s)
    backward_all (s)
    dense3.update_params()
    dense2.update_params()
    dense1.update_params()
    if n % 1000 == 0:
        print ("Step: ", n)
        mse = visualize2()


# In[ ]:



# In[203]:









# In[ ]:




