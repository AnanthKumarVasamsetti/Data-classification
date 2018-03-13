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
