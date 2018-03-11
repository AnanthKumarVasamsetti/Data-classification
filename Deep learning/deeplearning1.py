import numpy as np
import matplotlib.pyplot as plt
import h5py
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01

    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(W2.shape == (n_y, n_h))
    assert(b1.shape == (n_h, 1))
    assert(b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2    
    }

    return parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    #layer_dims is an containing number of hidden units at each layer    
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters["W"+str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layer_dims[l], 1))
    
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    #Used in backpropagation
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2    #Number of layers

    for l in range(1, L):
        A_prev = A
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]

        A, cache = linear_activation_forward(A_prev, W, b, "relu")

        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")

    caches.append(cache)

    #Assertion to check whether all the inputs are parsed or not
    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (-1./m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))

    cost = np.squeeze(cost)
    return cost

def main():
    Y, AL = compute_cost_test_case()

    print("cost = " + str(compute_cost(AL, Y)))

if __name__ == "__main__":
    main()