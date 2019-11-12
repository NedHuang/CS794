##########
# a4_20824226.py
##########
EPSILON = 0.01
MAX_ITERATION = 1000
#"train_data" and "train_label" are similar to the output of "svm_read_problem".
# A is a list of Dictionary, b is a list of int.
'''
b,A = svm_read_problem(os.getcwd()+'/a9a')
'''
# print('A is',type(A))
# print('b is',type(b))
# features = 123
# vec = DictVectorizer()
# A_matrix = A_matrix = vec.fit_transform(A).tocsr()
# b_matrix = csr_matrix(np.array(b).reshape(len(b),1), shape=(len(b),1))

# # shuffle matrix
# from sklearn.utils import shuffle
# A_shuffled, b_shuffled = shuffle(A_matrix,b_matrix)

# # Ax+beta = b, adding one column of one to A and append beta to x
# A = hstack((A_shuffled, csr_matrix(np.ones(shape=(A_shuffled.shape[0],1), dtype=float), shape=(len(b),1)))).tocsr()
# # x_ = csr_matrix(np.ones(shape=(features+1,1), dtype=float), shape=(features+1,1)) # features +1 = [x+beta]
# x0 = np.ones(shape=(features+1,1), dtype=float)

# # A is sparse matrix, x is ndarray, b is ndarray
# b = b_shuffled.toarray()
# # print('original types\t',type(A),type(x),type(b_))
# # print('original shapes\t',A.shape,x.shape,b.shape)

# # 90% training and 10% testing
# A_training = A[:int(0.9*A.shape[0])]
# b_training = b[:int(0.9*b.shape[0])]
# A_testing = A[int(0.9*A.shape[0]):]
# b_testing = b[int(0.9*b.shape[0]):]

# import 
import sys
import os
# add here your path to the folder libsvm-3.24/python
path = os.getcwd()+'/libsvm-3.24/python/'

print(os.getcwd())
# Add the path to the Python paths so Python can find the module.
sys.path.append(path)
# sys.path.append(os.getcwd()+'/libsvm-3.24/')
# sys.path.append(os.getcwd()+'/libsvm-3.24/python/')
print(path)


# Load the LIBSVM module.
from svmutil import *

# Add here your path to the folder libsvm-3.24
path = './libsvm-3.24/heart_scale'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# my import 
from numpy.linalg import norm
import math,time
from scipy import real, ndimage
from scipy.sparse import *
from sklearn.feature_extraction import DictVectorizer
from  scipy.sparse.linalg import expm
import random

# hinge loss
def get_hingeloss_nonsmooth(A ,x,b):
    A_dot_x = A.dot(x)
    loss_temp = A_dot_x*b
    loss = np.average(np.where(loss_temp < 1,1-loss_temp,0))
    return loss

# gradient of hinge loss
def get_hingeloss_nonsmooth_gradient(A,x,b):
    b_copy = np.copy(b)
    val = np.multiply(b_copy,A*x)
    b_copy[val > 1] = 0
    grad = A.multiply(-b_copy)
    grad = A.mean(0).transpose()
    return grad

# returns a n*1 nparray()
def get_hingeloss_nonsmooth_gradient_i(A,x,b,i):
    if 1 - (b[i]*(A[i].dot(x)))[0,0] > 0:
        grad = (-b[i]*A[i]).reshape(x.shape[0],1)
        return grad
    else: 
        grad = (np.zeros(shape=(x.shape[0],1), dtype=float))
        return grad

# stochastic_sub_gradient
def hingeloss_nonsmooth_stochastic_sub_gradient(A, x0, b, epsilon, max_iterations):
    x = x0
    loss = get_hingeloss_nonsmooth(A, x, b)
    loss_list = [loss]
    loss_min = loss
    loss_index = 0
    count = 0
    
    st = time.time()
    time_list = [st-st]
    grad = get_hingeloss_nonsmooth_gradient(A,x,b)
    while count < max_iterations:
#         step_size = 0.5/(count+1)
        step_size = 1/(count+1)
        # sample_i
        index = random.randint(0, A.shape[0]-1)
        g = get_hingeloss_nonsmooth_gradient_i(A,x,b,index )
        x = x - step_size * g
        
        
        if loss >1.05*loss_min:
            return x, loss_list, time_list
        loss_list.append(loss)
        loss = get_hingeloss_nonsmooth(A, x, b)
        count = count + 1
        time_list.append(time.time() - st)
    print('loss_min ',loss_min)
    return x, loss_list, time_list
vec = DictVectorizer()

class MyMethod:
    
    def __init__(self):
        self.A_training = None
        self.b_training = None
        self.x0 = None
        self.weight = None
        self.A_testing = None
        self.predicted_data = None
        self.scaled_pred_data = None
    
    def fit(self, train_data, train_label):
        self.A_training = vec.fit_transform(train_data).tocsr()
        self.b_training = np.array(train_label).reshape(len(train_label),1)
#       transfer A to sparse matrix and b to nparray. make x0 in proper size
        self.A_training = hstack((self.A_training, csr_matrix(np.ones(shape=(self.A_training.shape[0],1), dtype=float), shape=(self.A_training.shape[0],1)))).tocsr()
        self.x0 = np.ones(shape=(self.A_training.shape[1],1))
#         print(type(self.A_training),self.A_training.shape)
#         print(type(self.x0),self.x0.shape)
#         print(type(self.b_training),self.b_training.shape)
        self.weight,loss_list, time_list = hingeloss_nonsmooth_stochastic_sub_gradient(self.A_training, self.x0, self.b_training, EPSILON, MAX_ITERATION)
        # print(self.weight)
        print('finish fit')
        
    def predict(self, test_data):
        # print('--------predict')
        self.A_testing = vec.fit_transform(test_data).tocsr()
        # print(self.A_testing.shape)
        diff = self.A_training.shape[1] - self.A_testing.shape[1]
        # print('diff', diff)
        for i in range(diff):
            self.A_testing = hstack((self.A_testing, np.ones(shape=(self.A_testing.shape[0],1)))).tocsr()

        # print('new shape')
        # print(self.A_testing.shape)
        # print(type(self.A_testing),type(self.weight))
        # print(self.A_testing.shape,self.weight.shape)
        self.predicted_data = self.A_testing.dot(self.weight)

        self.scaled_pred_data = np.where(self.predicted_data>0,1,-1)
        return self.scaled_pred_data
'''  
obj = MyMethod()
obj.fit(A,b)
p = obj.predict(A)
'''

