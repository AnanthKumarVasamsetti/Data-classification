import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


np.random.seed(1) 

def test_Logistic_Regression(X, Y):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)
    
    #Plotting decision boundary for logistic regression
    plot_decision_boundary(lambda x:clf.predict(x), X, Y)   #Imported function
    
    plt.title("Logistic regression")
    plt.show()

"""
Arguments:
    X -- input dataset of shape(number of features, number of examples)
    Y -- output dataset of (output size, number of examples)
Returns:
    n_x --  Size of input layer
    n_h --  Size of hidden layer
    n_y --  Size of output layer
"""
def layer_sizes(X, Y):
    n_x = X.shape[0]    #No.of input layers
    n_h = 4             #No.of hidden layers
    n_y = Y.shape[0]    #No.of output layers

    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1":W1, "b1": b1, "W2":W2, "b2":b2}

    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"A1": A1, "Z1": Z1, "A2": A2, "Z2": Z2}

    return A2, cache

def main():
    X, Y = load_planar_dataset()
    plt.scatter(X[0,:], X[1, :], c = Y, s = 40, cmap=plt.cm.Spectral)
    #This is for testing logistic regression
    #test_Logistic_Regression(X, Y)
    X_assess, parameters = forward_propagation_test_case()

    A2, cache = forward_propagation(X_assess, parameters)

    # Note: we use the mean here just to make sure that your output matches ours. 
    print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
if __name__ == '__main__':
    main()