#!/usr/bin/env python
'''
This script implements logistic regression on train data from HW1, get train accuracy
Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/19/2022
'''
from utilities import Load_data_logisticreg_train, Load_data_logisticreg_test
import numpy as np
from utilities import get_gradient
import random
random.seed(75)
np.random.seed(75)

str_path_1c_test='./data_test_2a3a.txt'
str_path_1b_train = './data_2c2d3c3d_program.txt'
# load train data from HW1
arrTrainX,arrTrainY = Load_data_logisticreg_train(str_path_1b_train,remove_age=False)
arrTrainY = arrTrainY.ravel()
arrTrainY=arrTrainY.reshape([120,1])
# load test data from Hw1
arrTestX = Load_data_logisticreg_test(str_path_1c_test, remove_age=False)

#initialize theta and alpha
theta = np.array([1.0,1.0,1.0],dtype=float)
alpha = 0.0001
pred_y = []
epochs = 2000

for epoch in range(epochs):
    arrconcat = np.concatenate((arrTrainX,arrTrainY),axis = 1)
    np.random.shuffle(arrconcat)
    arrTrainX_shuffle = arrconcat[:,:-1]
    arrTrainY_shuffle = arrconcat[:, -1]
    for i in range(arrTrainX_shuffle.shape[0]):
        gradient = get_gradient(arrTrainX_shuffle[i,:],arrTrainY_shuffle[i,],theta)
        theta = theta+np.dot(alpha,gradient)

#predict y

pred_y_test = np.dot(arrTestX,theta)
logit_y_test = [1 / (1 + (np.exp(-x))) for x in pred_y_test]
y_test_prob = np.round(logit_y_test,0)
pred_y_train = np.dot(arrTrainX,theta)
logit_y_train = [1 / (1 + (np.exp(-x))) for x in pred_y_train]
y_train_prob = np.round(logit_y_train,0)

#calculate train accuracy
counter = 0
for i in range(len(y_train_prob)):
    if (y_train_prob[i] == arrTrainY[i]):
        counter = counter + 1
accuracy = (counter / len(y_train_prob)) * 100
print('logistic regression --- SGD --- Train Accuracy : '+str(accuracy))
