#!/usr/bin/env python
'''
This script implements logistic regression using SGD on train data from HW1 without age feature. Performance is evaluated using leave one out validation
Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/19/2022
'''
from utilities import Load_data_logisticreg_train, Load_data_logisticreg_test
import numpy as np
from utilities import get_gradient
import random
random.seed(445)
np.random.seed(445)

str_path_1b_train = 'data_2c2d3c3d_program.txt'

# load train data from HW1
arrTrainX,arrTrainY = Load_data_logisticreg_train(str_path_1b_train,remove_age=True)
arrTrainY = arrTrainY.ravel()


#initialize theta and alpha
theta = np.array([1.0,1.0],dtype=float)
alpha = 0.0001
pred_y = []
epochs = 6000

predictions = []
for j in range(arrTrainX.shape[0]):
    test_sample = arrTrainX[j,:]
    arrTrainX_tmp = np.delete(arrTrainX,j,0)
    arrTrainY_tmp = np.delete(arrTrainY,j)
    arrTrainY_tmp = arrTrainY_tmp.reshape([arrTrainY_tmp.shape[0],1])
    for epoch in range(epochs):
        arrconcat = np.concatenate((arrTrainX_tmp, arrTrainY_tmp), axis=1)
        np.random.shuffle(arrconcat)
        arrTrainX_shuffle = arrconcat[:, :-1]
        arrTrainY_shuffle = arrconcat[:, -1]
        for i in range(arrTrainX_shuffle.shape[0]):
            gradient = get_gradient(arrTrainX_shuffle[i, :], arrTrainY_shuffle[i,], theta)
            theta = theta + np.dot(alpha, gradient)
    #predict y
    pred_y_test = np.dot(test_sample,theta)
    logit_y_test = 1 / (1 + (np.exp(-pred_y_test)))
    y_test_prob = np.round(logit_y_test,0)
    predictions.append(y_test_prob)

#calculate train accuracy
counter = 0
for i in range(len(predictions)):
    if (predictions[i] == arrTrainY[i]):
        counter = counter + 1
accuracy = (counter / len(predictions)) * 100

print('logistic reg LOO accuracy is: '+str(accuracy))
