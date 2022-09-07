#!/usr/bin/env python
'''
This script implements logistic regression on train data from HW1, using 2d projections of training data(pca) and plots the decision boundary
Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/19/2022
'''
from utilities import Load_data_logisticreg_train, Load_data_logisticreg_test
import numpy as np
from utilities import get_gradient
import matplotlib.pyplot as plt
import random
random.seed(5)
np.random.seed(5)

str_path_1b_train = './data_2c2d3c3d_program.txt'

# load train data from HW1
arrTrainX,arrTrainY = Load_data_logisticreg_train(str_path_1b_train,remove_age=False)
X = arrTrainX - arrTrainX.mean(axis=0)
# Normalize
Z = X / arrTrainX.std(axis=0)
#computing covariance matrix
Z = np.dot(Z.T, Z)
eigenvalues, eigenvectors = np.linalg.eig(Z)
D = np.diag(eigenvalues)
P = eigenvectors
print(D)
#choosing 2 principle components based on highest eigen values
pc = np.dot(arrTrainX,P[:,1:3])
print(pc)

#constructing 2D space of principle comonents
test1 = np.linspace(-70,-30, 20000)
np.random.shuffle(test1)
test2 = np.linspace(10,40,20000)
np.random.shuffle(test2)

test1 = test1.reshape(20000,1)
test2 = test2.reshape(20000,1)

testpc = np.concatenate((test1,test2),axis = 1)
arrTrainX =pc

arrTrainY = arrTrainY.ravel()
arrTrainY=arrTrainY.reshape([120,1])

#initialize theta and alpha
theta = np.array([1.0,1.0],dtype=float)
alpha = 0.001
pred_y = []
epochs = 1000

for epoch in range(epochs):
    arrconcat = np.concatenate((arrTrainX,arrTrainY),axis = 1)
    np.random.shuffle(arrconcat)
    arrTrainX_shuffle = arrconcat[:,:-1]
    arrTrainY_shuffle = arrconcat[:, -1]
    for i in range(arrTrainX_shuffle.shape[0]):
        gradient = get_gradient(arrTrainX_shuffle[i,:],arrTrainY_shuffle[i,],theta)
        theta = theta+np.dot(alpha,gradient)

#predict y

#pred_y_test = np.dot(arrTestX,theta)
#logit_y_test = [1 / (1 + (np.exp(-x))) for x in pred_y_test]
#y_test_prob = np.round(logit_y_test,0)
pred_y_train = np.dot(arrTrainX,theta)
logit_y_train = [1 / (1 + (np.exp(-x))) for x in pred_y_train]
y_train_prob = np.round(logit_y_train,0)

## predict PC

pred_y_trainpc = np.dot(testpc,theta)
logit_y_trainpc = [1 / (1 + (np.exp(-x))) for x in pred_y_trainpc]
y_train_probpc = np.round(logit_y_trainpc,2)


idx = np.where(y_train_probpc == 0.50)

# get pcs where logits are 0.5.This is the decision boundary
boundaryxy = testpc[idx]
bound = boundaryxy.reshape(boundaryxy.shape[0],boundaryxy.shape[1])

#calculate train accuracy
counter = 0
for i in range(len(y_train_prob)):
    if (y_train_prob[i] == arrTrainY[i]):
        counter = counter + 1
accuracy = (counter / len(y_train_prob)) * 100
print('logistic regression --- SGD --- Train Accuracy : '+str(accuracy))

#plt.plot(test1,test2,'*b')
#plt.plot(pc[:,0],pc[:,1], '*b')
plt.scatter(pc[:,0],pc[:,1], s=100, c=arrTrainY)
plt.plot(bound[:,0],bound[:,1],'-r')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title('scatter plot of 2 principle components and decision boundary')
plt.show()
