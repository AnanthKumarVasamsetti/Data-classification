"""
This program implements a two layer neural network
Architecture is as follows:
    --> initialize_parameters:
            Here number of input units, hidden units and output units are generated according to user requirement
    --> linear_activation_forward:
            Given data is travelled towards output unit and predicts
    --> compute_cost:
            Once output is generated cost computation is taken place, it shows for a given parameters set what is the loss in prediction
    --> linear_activation_backward:
            Here the algorithm travels towards input layer to find the deviations of weights from the given output
    --> update_parameters:
            After gathering deviations for each parameter, they are updated using learning rate

This process is continued for number of times and cost is observed at each stage, the lesser the cost the better the model state
"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

def initialize_paramters(n_x, n_h, n_y):
        W1 = np.random.randn(n_h, n_x) * 0.01
        W2 = np.random.randn(n_y, n_h) * 0.01

        b1 = np.zeros((n_h, 1))
        b2 = np.zeros((n_y, 1))

        parameters = {
                "W1" : W1,
                "W2" : W2,
                "b1" : b1,
                "b2" : b2
        }

        return parameters

def forward_propagation(W, A_prev, b, activation_function):
    Z = np.dot(W, A_prev) + b

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    linear_cache = (W, A_prev, b)

    if (activation_function == "relu"):
        A = np.maximum(0, Z)

    elif (activation_function == "sigmoid"):
        A = 1/(1+np.exp(-Z))

    assert(A.shape == Z.shape)

    activation_cache = Z

    cache = (linear_cache, activation_cache)

    return A, cache

def compute_cost(A, Y):
    m = Y.shape[1]

    cost = (-1./m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A)))
    cost = np.squeeze(cost)

    return cost

def backward_propagation(dA, cache, activation_function):
    linear_cache = cache[0]
    activation_cache = cache[1]
    z = activation_cache

    m = dA.shape[1]
    W = linear_cache[0]
    A = linear_cache[1]
    b = linear_cache[2]

    #calculating g'(z)
    if (activation_function == "sigmoid"):
        s = 1/(1 + np.exp(-z))
        dz = dA * s * (1 - s)

    elif (activation_function == "relu"):
        dz = np.array(dA, copy = True)
        dz[z <= 0 ] = 0

    assert(dz.shape == z.shape)

    dW = 1/m * np.dot(dz, A.T)
    db = 1/m * np.sum(dz, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dz)

    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] = parameters["W1"] - learning_rate * grads["W1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["W2"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["b1"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["b2"]

    return parameters

def two_layer_model(X, y, layer_dims, learning_rate, num_iterations, print_cost = True):
        parameters = initialize_parameters(layer_dims[0], layer_dims[1], layer_dims[2])

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        costs = []
        grads = {}

        for i in range(0, num_iterations):
                #Forward propagation
                A1, cache1 = forward_propagation(W1, X, b1, "relu")
                A2, cache2 = forward_propagation(W2, A1, b2, "sigmoid")

                #Cost computation
                cost = compute_cost(A2, y)

                dA2 = -(np.divide(y, A2) - np.divide(1-y, 1-A2))

                dA1, dW2, db2 = backward_propagation(dA2, cache2, "sigmoid")
                dA0, dW1, db1 = backward_propagation(dA1, cache1, "relu")

                grads["W1"] = dW1
                grads["W2"] = dW2
                grads["b1"] = db1
                grads["b2"] = db2

                parameters = update_parameters(parameters, grads, learning_rate)

                W1 = parameters["W1"]
                W2 = parameters["W2"]
                b1 = parameters["b1"]
                b2 = parameters["b2"]

                if (i%100 == 0):
                    print("Cost at iteration "+str(i)+": "+str(cost))
                    costs.append(cost)

        plt.plot(costs)
        plt.ylabel('Costs')
        plt.xlabel('Iterations')
        plt.show()
def main():
        train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

        #Reshaping the training and testing samples
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        train_x = train_x_flatten / 255
        test_x = test_x_flatten / 255

        two_layer_model(train_x, train_y, [train_x.shape[0], 7, 1], 0.0075, 10000, True)

if __name__ == "__main__":
        main()
