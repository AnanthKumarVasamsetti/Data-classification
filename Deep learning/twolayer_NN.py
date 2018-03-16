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

def two_layer_model(X, y, layer_dims, learning_rate, num_iterations, print_cost = True):
        parameters = initialize_parameters(layer_dims[0], layer_dims[1], layer_dims[2])

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        print(parameters)
def main():
        train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

        #Reshaping the training and testing samples
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        train_x = train_x_flatten / 255
        test_x = test_x_flatten / 255

        two_layer_model(train_x, train_y, [train_x.shape[0], 7, 1], 0.0075, 3000, True)

if __name__ == "__main__":
        main()