"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd
import numpy.linalg as alg


###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    ma_error = None
    pred = X.dot(w)
    ma_error = (abs(pred - y)).mean()
    return ma_error

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################
  weight = None
  assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
  weight = alg.inv(X.T.dot(X)).dot(X.T).dot(y)
  return weight

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    weight = None
    mtx = X.T.dot(X)
    temp = 0.1*np.identity(mtx.shape[0])
    eig_val, eig_vec = alg.eig(mtx)
    while np.abs(eig_val).min() < 0.00001:
        mtx += temp
        eig_val, eig_vec = alg.eig(mtx)
    weight = alg.inv(mtx).dot(X.T).dot(y)
    return weight


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################
    weight = None
    mtx = X.T.dot(X)
    temp = lambd * np.identity(mtx.shape[0])
    mtx += temp
    weight = alg.inv(mtx).dot(X.T).dot(y)
    return weight

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - best_lambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    best_lambda = None
    best_abs_error = 99999.9
    for k in range(-19, 20):
        lamb = 10**k
        weight = regularized_linear_regression(Xtrain, ytrain, lamb)
        abs_error = mean_absolute_error(weight, Xval, yval)
        if abs_error < best_abs_error:
            best_lambda = lamb
            best_abs_error = abs_error
    return best_lambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    pow_x = [X]
    for k in range(1, power):
        pow_x.append(pow_x[k-1]*X)
    return np.concatenate(pow_x, axis=1)
