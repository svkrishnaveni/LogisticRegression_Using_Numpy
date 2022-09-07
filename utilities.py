#!/usr/bin/env python
'''
This script contains various functions used in this project
Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/15/2022
'''

import numpy as np
import math
import re
import matplotlib.pyplot as plt
import os

########################################## Load data functions################################
# function to load data (train or test) separated as features and targets and generate derived features of x based on depth (d) and k
def Load_data(strpath_todata,d,k):
    '''
    This function populates feature set based on input depth 'd' in to numpy array
    inputs: str path to train_data.txt or test_data.txt, integer value of depth(d), integer value of frequency increment(k)
    outputs: numpy array of feature set,numpy arrays of targets and features
    '''
    # initialize empty lists to gather features and targets
    features = []
    targets = []
    # read lines in txt file as string
    with open(strpath_todata) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 1 feature
            lsFeature_tmp = [float(data_tmp.split(',')[0])]
            # extract target
            lsTarget_tmp = [float(data_tmp.split(',')[1])]
            features.append(lsFeature_tmp)
            targets.append(lsTarget_tmp)
    features = np.array(features)
    targets = np.array(targets)
    #determining the number of features for given depth d
    num_of_features = 2*(d+1)
    #Initializing array for feature set
    arr_x = np.zeros((len(features),num_of_features))
    #populating first column in feature set array with ones
    arr_x[:,0]=1
    #populating second column in feature set with x
    arr_x[:,1]=features[:,0]
    #populating remaining columns in feature set array
    z=2
    for i in range(d):
        a=[]
        c=[]
        i=i+1
        for j in range(len(features)):
            b = math.sin(i*k*features[j])
            a.append(b)
        arr_a = np.array(a)
        arr_x[:,z] = arr_a
        for j in range(len(features)):
            d = math.cos(i*k*features[j])
            c.append(d)
        arr_c = np.array(c)
        arr_x[:,z+1] = arr_c
        z=z+1+1
    #coverting list of features into array of features
    X= [x[0] for x in features]
    X = np.array(X)
    #coverting list of targets into array of targets
    targets = [x[0] for x in targets]
    targets = np.asarray(targets)
    #returning features set for a given depth 'd',train targets,input X
    return arr_x ,targets, X

#Loading program data separated as features and targets for d=0
def Load_data_2(strpath_todata):
    '''
    This function loads data from given input path(strpath_todata) then extracts x and labels
    and populates feature set with x and intercept
    Inputs : path to input text file
    Outputs : numpy arrays of feature_set,targets and x
    '''
    # initialize empty lists to gather features and targets
    features = []
    targets = []
    # read lines in txt file as string
    with open(strpath_todata) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 1 feature
            lsFeature_tmp = [float(data_tmp.split(',')[0])]
            # extract target
            lsTarget_tmp = [float(data_tmp.split(',')[1])]
            features.append(lsFeature_tmp)
            targets.append(lsTarget_tmp)
    features = np.array(features)
    targets = np.array(targets)
    #Initializing array for feature set with 2 columns
    arr_x = np.zeros((len(features),2))
    #populating first column in feature set array of ones
    arr_x[:,0]=1
    #populating second column in feature set array with x
    arr_x[:,1]=features[:,0]
    #coverting list of features into array of features
    X = [x[0] for x in features]
    X = np.array(X)
    #coverting list of targets into array of targets
    targets = [x[0] for x in targets]
    targets = np.asarray(targets)
    #returning features set(chosen hypothesis space),train targets,train features
    return arr_x ,targets, X


# Loading homework1 train data separated as features and targets and modifying features into feature set
def Load_data_logisticreg_train(str_path_1b_program, remove_age=False):
    '''
    This function loads train data(demographic data height,weight,age) from homework1 and populates feature set
    inputs: str path to train data.txt
    outputs: numpy array of feature set,numpy arrays of targets,features
    '''
    # initialize empty lists to gather features and targets
    features = []
    targets = []
    # read lines in txt file as string
    with open(str_path_1b_program) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 1 feature
            lsFeature_tmp = [float(data_tmp.split(',')[0]), float(data_tmp.split(',')[1]), int(data_tmp.split(',')[2])]
            # extract target
            lsTarget_tmp = [data_tmp.split(',')[3][1]]
            features.append(lsFeature_tmp)
            targets.append(lsTarget_tmp)
        if remove_age:
            for i in features:
                del i[2]
    features = np.array(features)
    targets = np.array(targets)
    features_col_len = features.shape[1]
    # Initializing array for feature set with 2 columns
    arr_x = np.zeros((len(features), features_col_len + 1))
    # populating first column in feature set array as 1
    arr_x[:, 0] = 1
    # populating remaining columns in feature set array
    for i in range(features_col_len):
        arr_x[:, i + 1] = features[:, i]
    # coverting list of features into array of features
    features = [x[:] for x in features]
    features = np.asarray(features)
    # coverting list of targets into array of targets
    targets = [x[:] for x in targets]
    targets = np.asarray(targets)
    # converting targets from string to number
    target_labels =np.zeros([len(targets),1])
    for i in range(len(targets)):
        if(targets[i]=='W'):
            target_labels[i] = 0
        else:
            target_labels[i] = 1
    target_labels= target_labels.astype(float)


    # returning features set(chosen hypothesis space),train targets,train features
    return features,target_labels

#Loading test data with features from HW1
def Load_data_logisticreg_test(str_path_2a3a_test,remove_age = False):
    '''
    This function loads data(demographic data height,weight,age) from homework1 and populates feature set
    inputs: str path to train data.txt
    outputs: numpy array of feature set,numpy arrays of targets,features
    '''
    # initialize empty lists to gather features and targets
    features = []
    # read lines in txt file as string
    with open(str_path_2a3a_test) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 3 features
            lsFeature_tmp = [float(data_tmp.split(',')[0]),float(data_tmp.split(',')[1]),int(data_tmp.split(',')[2])]
            features.append(lsFeature_tmp)
        features = np.array(features)
    if remove_age:
            for i in features:
                del i[2]

    features_col_len = features.shape[1]
    #Initializing array for feature set with 2 columns
    arr_x = np.zeros((len(features),features_col_len+1))
    #populating first column in feature set array as 1
    arr_x[:,0]=1
    #populating remaining columns in feature set array
    for i in range(features_col_len):
        arr_x[:,i+1]=features[:,i]
    #coverting list of features into array of features
    features = [x[:] for x in features]
    features = np.asarray(features)
    #returning features set(chosen hypothesis space),train targets,train features

    return features


########################################## Functions for Linear regression ################################
#Linear Regression
def fit_lin_reg(features,targets):
    '''
    This function identifies coefficients using analytic approach of penrose moore's pseudo inverse
    inputs: numpy array of features and corresponding targets
    outputs: numpy array of parameters/coefficients
    '''
    #identifying parameters(coefficients) using analytic optimization
    feature_transpose = features.transpose()
    feature_set = np.dot(feature_transpose,features)
    inv_feature_set = np.linalg.inv(feature_set)
    feature_set_mul = np.dot(inv_feature_set,feature_transpose)
    coefficients = np.dot(feature_set_mul,targets)
    return coefficients

def predict(feature_set, coefficients):
    '''
    This function predicts target using coefficients on given feature_set (either train or test)
    inputs:input array of feature_set (train or test), trained coefficients
    outputs: predicted targets for given data
    '''
    pred_targets = np.dot(feature_set, coefficients)
    return np.array(pred_targets)

# plot models for given function depth d
def plot_1b(str_path_1b_traindata, d, k, save_figures=False):
    '''
    This function plots and saves figures for estimated function of the signal [bestfit] for a given depth 'd' together with original  data points
    inputs: integer value of depth(d), integer value of frequency increment(k), boolean save_figures - saves plots in current dir
    outputs:
    '''
    for i in range(d+1):
        arr_features_set, targets, x = Load_data(str_path_1b_traindata, i, k)
        # sort x - get indices
        x_sorted = np.argsort(x)
        coef = fit_lin_reg(arr_features_set, targets)
        # getting predicted values for given data points
        y_pred = predict(arr_features_set, coef)
        # plotting given features with given targets
        plt.plot(x[x_sorted], targets[x_sorted], '*g')
        # plotting given features with predicted targets
        plt.plot(x[x_sorted], y_pred[x_sorted], '-b')
        # giving labels to x and y axis
        plt.xlabel('x')
        plt.ylabel('signal amplitude')
        plt.legend(['original signal', 'bestfit'])
        # giving title to the plotted graph
        plt.title('discrete signal points and best fit with function depth = ' + str(i))
        # saving the plots in curent dir
        if save_figures:
            if not os.path.isdir('./plots'):
                os.mkdir('./plots')
                plt.savefig('./plots/Bestfit_d_' + str(i) + '.png', dpi=150)

        plt.show()
        plt.clf()


########################################## Functions for locally weighted Linear regression ################################

def plot_2b(str_path_2b_traindata, save_figures = False):

    arr_features_set, targets, arr1D_train_x = Load_data_2(str_path_2b_traindata)
    gamma = 0.1

    # sort x - get indices
    x_sorted = np.argsort(arr1D_train_x)

    # predicting targets for each test feature
    pred_targets =[]
    for i in range(len(arr1D_train_x)):
        test_feature = arr1D_train_x[i]
        weights = get_weights(arr1D_train_x,test_feature,gamma)
        coefficients = get_coef_local_lin_reg(weights,arr_features_set,targets)
        # predicting y
        t = np.dot(coefficients, arr_features_set[i])
        pred_targets.append(t)
    # getting predicted values for given data points
    pred_targets = np.array(pred_targets)

    # plotting given features with given targets
    plt.plot(arr1D_train_x[x_sorted], targets[x_sorted], '*g')
    # plotting given features with predicted targets
    plt.plot(arr1D_train_x[x_sorted], pred_targets[x_sorted], '-b')
    # giving labels to x and y axis
    plt.xlabel('x')
    plt.ylabel('signal amplitude')
    plt.legend(['original signal', 'LocalLinReg'])
    # giving title to the plotted graph
    plt.title('Locally weighted Lin reg function with original datapoints')
    # saving the plots in curent dir
    if save_figures:
        if not os.path.isdir('./plots'):
            os.mkdir('./plots')
            plt.savefig('./plots/Bestfit_localLinReg' + '.png', dpi=150)

    plt.show()
    plt.clf()


# function for computing mean squared error
def mse(y_pred, y):
    y_pred = np.asarray(y_pred)
    y = np.asarray(y)
    assert (y_pred.shape == y.shape), 'y_pred and y have different lengths!'
    error = y_pred - y
    squared_error = np.square(error)
    mean_squared_error = squared_error.mean(axis=0)
    return mean_squared_error

def get_coef_local_lin_reg(weights,arr2D_feature_set,arr1D_targets):
    '''
    This function computes optimized coefficients for a local input using gaussian neighboring weights
    inputs: numpy array of weights,training feature_set and training labels
    outputs: numpy arrays of coefficients
    '''
    #diagonalizing array weights
    arrW_test_diag = np.diag(weights)
    # computing locally weighted lin.reg model for the test_datapoint to get best parameters
    arr2D_feature_set_transpose = arr2D_feature_set.transpose()
    f_set_trans_mul_weight = np.dot(arr2D_feature_set_transpose,arrW_test_diag)
    f_set_trans_mul_weight_mul_arr2D_feature_set = np.dot(f_set_trans_mul_weight,arr2D_feature_set)
    inv_f_set_trans_mul_weight_mul_arr2D_feature_set = np.linalg.inv(f_set_trans_mul_weight_mul_arr2D_feature_set)
    f_set_trans_mul_weight_mul_arr1D_targets = np.dot(f_set_trans_mul_weight,arr1D_targets)
    #determine=ing optimized coefficients
    coefficients = np.dot(inv_f_set_trans_mul_weight_mul_arr2D_feature_set,f_set_trans_mul_weight_mul_arr1D_targets)
    return coefficients


#Calculates weights for each train feature w.r.t given test feature
def get_weights(train_features,test_feature,gamma):
    '''
    This function computes weights of train points by gaussian weighting w.r.t given test point
    Inputs : numpy array of train_features,single test point,gamma value
    OUtputs : numpy array of weights
    '''
    weights = []
    for i in range(len(train_features)):
        diff_sq = np.square(train_features[i]-test_feature)
        w = np.exp(-(diff_sq)/(2*np.square(gamma)))
        weights.append(w)
    weights = np.asarray(weights)
    return weights


######################################## Functions for SGD in Logistic regression #################################
def get_gradient(x,y,theta):
    theta_trans = np.transpose(theta)
    theta_trans_mul_x = np.dot(theta_trans,x)
    logit =  1 / (1 + (np.exp(-theta_trans_mul_x)))
    gradient = np.dot((y - logit),x)
    return gradient


