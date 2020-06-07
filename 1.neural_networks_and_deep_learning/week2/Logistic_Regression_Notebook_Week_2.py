#!/usr/bin/env python
# coding: utf-8

# # 1. Load Packages

# In[138]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from PIL import Image
from scipy import ndimage

get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Load Data
# The coursera notebook has a customized module lr_utils to load dataset. But that is not available for everyone. So I have downloaded the dataset from the coursera website and now I will be importing that dataset.

# In[139]:


train_set_x = pd.read_csv('data/train_set_x.csv')
train_set_y = pd.read_csv('data/train_set_y.csv')
test_set_x = pd.read_csv('data/test_set_x.csv')
test_set_y = pd.read_csv('data/test_set_y.csv')


# In[140]:


for data in [train_set_x,train_set_y,test_set_x,test_set_y]:
    data.drop('Unnamed: 0', axis = 1, inplace = True)


# In[141]:


print(train_set_x.shape, train_set_y.shape, test_set_x.shape, test_set_y.shape)


# In[142]:


train_set_x = np.array(train_set_x)
train_set_y = np.array(train_set_y)
test_set_x = np.array(test_set_x)
test_set_y = np.array(test_set_y)


# In[143]:


def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a


# In[144]:


def initilize_parameters(dim):
    W = np.zeros(dim) #dimension in this case will be - (n_px*n_px*3,1)
    b = 0
    return W, b


# In[145]:


def propagation(W, b, X, Y):
    '''finding z, a, L, J, dw, db 
    required - w,b, X, Y'''
    m = X.shape[1]
    Z = np.dot(W.T,X) + b #shape (1,m)
    A = sigmoid(Z) #shape (1,m)
    cost = (1/m)*np.sum(-(Y)*np.log(A) - (1-Y)*(np.log(1-A))) #cost function = average of all loss function.
    # we are using normal product (element-wise) because Y and log(A) both have the same dimension and both of them are a number for a training example
    
    dZ = A-Y
    dW = np.dot(X, dZ.T)/m  #dJ/dw = dw = x*dz --> X*(A-Y)
    db = dZ.sum()/m # shape (1,1) 
    
    grad = {'dW':dW, 'db':db}
    return grad, cost


# In[204]:


def optimization(W, b, X, Y, num_iteration, alpha, print_cost = False):
    costs = []
    for i in range(num_iteration):
        
        # gradients
        grads, cost = propagation(W,b,X,Y)
        dW = grads['dW']
        db = grads['db']

        # update parameters
        W = W - alpha*dW
        b = b - alpha*db
        
        # print cost for every 100th iteration
        if i%1000 == 0:
            costs.append(cost)
        if print_cost == True and i%1000== 0:
            print(f"cost after {i}th iteration:{cost}")
    
    params = {'W':W, 'b':b}
    grads = {'dW':dW, 'db':db}
    
    return params, grads, costs 


# In[205]:


W, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagation(W, b, X, Y)
print ("dW = " + str(grads["dW"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


# In[206]:


params, grads, costs = optimization(W, b, X, Y, num_iteration= 100, alpha= 0.009, print_cost = False)

print ("W = " + str(params["W"]))
print ("b = " + str(params["b"]))
print ("dW = " + str(grads["dW"]))
print ("db = " + str(grads["db"]))


# In[207]:


params, grads, cost = optimization(W, b, X, Y, 1000, 0.3, print_cost = True)


# In[208]:


def predict(W, b, X):
#     W = W.reshape(X.shape[1],1)
    A = sigmoid(np.dot(W.T, X) + b) #the predictions - shape - (1,m)
    y_pred = np.zeros((1,A.shape[1]))
    
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            y_pred[0,i]=1
        else:
            y_pred[0,i] = 0
    
    return A, Z, y_pred


# In[209]:


def model(train_x, train_y, test_x, test_y, num_iteration, alpha, print_cost = False):
    #training our model
    #intitialize parameters:
    m = train_x.shape[1]
    n = train_x.shape[0]
    W,b = initilize_parameters((n,1))
    
    #forward propagation
    grad, cost = propagation(W,b,train_x, train_y)
    
    #extract gradients
    dW = grad['dW']
    db = grad['db']
    
    #back propagation:
    params, grads, costs = optimization(W,b,train_x, train_y,  num_iteration, alpha, print_cost)
    
    #extract parameters and gradients
    W = params['W']
    b = params['b']
    dW = grads['dW']
    db = grads['db']
    
    # make prediction:
    A, Z, y_predicted_train = predict(W, b, train_x)
    A, Z, y_predicted_test = predict(W,b, test_x)
    
    #accuracy:
    differences_train = (y_predicted_train - train_y)
    differences_test = (y_predicted_test - test_y)
    
    correct_prediction_train = (differences_train==0).sum() #number of zeros in differences_train
    correct_prediction_test = (differences_test==0).sum() #number of zeros in differences_test
    
    accuracy_train = correct_prediction_train/train_y.shape[1]
    accuracy_test = correct_prediction_test/test_y.shape[1]
    
    print(accuracy_train, accuracy_test)
    return y_predicted_train, y_predicted_test, A, Z, differences_train, differences_test


# In[211]:


model(train_set_x,train_set_y,test_set_x, test_set_y,10000,0.003, True)


# The model performs good on training set as the accuracy is around 98%. But the model accuracy on test set is the worst. So, this is the case of overfitting. In the next week's notebook, we will learn concepts like regularisation and hyperparametric tuning to resolve this issue.
