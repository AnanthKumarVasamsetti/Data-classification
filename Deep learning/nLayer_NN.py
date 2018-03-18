import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

def initialize_paramters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    np.random.seed(1)

    for i in range(1, L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))

        assert(parameters["W"+str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        assert(parameters["b"+str(i)].shape == (layer_dims[i], 1))

    return parameters


def forward_propagation(W, A_prev, b, activation_function):
    Z = np.dot(W, A_prev) + b
    linear_cache = (W, A_prev, b)

    if(activation_function == "relu"):
        A = 1/(1 + np.exp(-Z))

    elif(activation_function == "sigmoid"):
        A = np.maximum(0, Z)

    activation_cache = Z

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []

    for l in range(1, L):
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        A_prev = A

        A, cache = forward_propagation(W, A_prev, b, "relu")
        caches.append(cache)

    AL, cache = forward_propagation(parameters["W"+str(L)], A, parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

def compute_cost(AL, y):
    m = y.shape[1]

    cost = (-1./m) * np.sum(np.multiply(y, np.log(AL)) + np.multiply((1-y), np.log(1 - AL)))
    cost = np.squeeze(cost)

    return cost

def backward_propagation(dA, cache, activation_function):
    linear_cache = cache[0]
    activation_cache = cache[1]

    Z = activation_cache
    m = dA.shape[1]

    W = linear_cache[0]
    A = linear_cache[1]
    b = linear_cache[2]

    if(activation_function == "sigmoid"):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * (s * (1 - s))

    elif(activation_function == "relu"):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    dW = 1/m * np.dot(dZ, A.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def L_model_backward(A, y, caches):
    m = y.shape[1]
    y = y.reshape(A.shape)
    L = len(caches)
    grads = {}

    dA = -(np.divide(y, A) - np.divide((1-y), (1-A)))
    current_cache = caches[L - 1]

    dA_prev, dW, db = backward_propagation(dA, current_cache, "sigmoid")

    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA = dA_prev

        dA_prev, dW, db = backward_propagation(dA, current_cache, "relu")

        grads["dW"+str(l + 1)] = dW
        grads["db"+str(l + 1)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

def n_layer_model(X, y, layer_dims, learning_rate, num_of_iterations, print_cost):
    m = y.shape[1]

    parameters = initialize_paramters(layer_dims)

    for i in range(0, num_of_iterations):

        #Forward propagation
        AL, caches = L_model_forward(X, parameters)

        #Compute cost
        cost = compute_cost(AL, y)

        #Backward propagation
        grads = L_model_backward(AL, y, caches)

        #Updating
        parameters = update_parameters(parameters, grads, learning_rate)

        if(print_cost and i%100 == 0):
            print("Cost at iteration "+str(i)+": "+str(cost))

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    #Reshaping the training and testing samples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    n_layer_model(train_x, train_y, [train_x.shape[0], 20, 7, 5, 1], 0.075, 5000, True)

if __name__ == "__main__":
    main()
