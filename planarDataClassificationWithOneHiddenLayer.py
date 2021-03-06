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

Description:
    Assign size for each layers gives a picture of neural network and intution of it's working
"""
def layer_sizes(X, Y):
    n_x = X.shape[0]    #No.of input layers
    n_h = 4             #No.of hidden layers
    n_y = Y.shape[0]    #No.of output layers

    return (n_x, n_h, n_y)
"""
Arguments:
    n_x --  number of input layers
    n_h --  number of hidden layers
    n_y -- number of output layers

Returns:
    parameters  --  JSON of randomly initialized weights for each layer

Description:
    Random initialization of weights are better than assiging zeros as it reduces the weight adjust iterations
    to some extent
"""
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

"""
Arguments:
    X           --  Data matrix where the column represents each example and each row represents feature of that example
    parameters  --  A JSON containing the weights and intercepts

Returns:
    A2      --  A prediction data or output matrix whose column number represents the number of examples
    cache   --  A JSON containing data that it attained at each layer and forwarded to it's successor layer

Description:
    Based on the adjusted and assigned weights at each layer forward propagation is carried out to provide output
    for each input data
"""
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

"""
Arguments:
    A2  --  A prediction data or output matrix whose column number represents the number of examples
    Y   --  Label matrix which consists the actual output of the given input data

Returns:
    cost    --  A floating integer that results how much the predicted data is accurate

Description:
    Cost calculation is carried out by cross validatiing between expected output and actual output using cost-entropy function
"""
def compute_cost(A2, Y):
    m = A2.shape[1]
    #Cost-entropy function
    log_probs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    
    cost = -1/m * np.sum(log_probs)
    cost = np.squeeze(cost)
    
    return cost

"""
COME BACK TO THIS IT IS TOATALLY ANNOYING YOU TO UNDERSTAND THE WHOLE CRUX
"""
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W1 = parameters["W1"]
    b1 = parameters["b1"]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1":dW1, "db1": db1, "dW2":dW2, "db2":db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1":W1, "b1": b1, "W2":W2, "b2":b2}

    return parameters

def neural_network_model(X, Y, n_h, num_iterations, print_cost = True):
    np.random.seed(3)

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X , parameters)

        cost = compute_cost(A2, Y)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if(print_cost and i%1000 == 0):
            print("Cost after "+str(i)+" iteration: "+str(cost))
    
    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

def main():
    
    X, Y = load_planar_dataset()
    plt.scatter(X[0,:], X[1, :], c = Y, s = 40, cmap=plt.cm.Spectral)
    #This is for testing logistic regression
    #test_Logistic_Regression(X, Y)
    
    #===================================================================================================
    parameters = neural_network_model(X, Y, n_h = 10, num_iterations = 10000, print_cost=True)
    predictions = predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
    #===================================================================================================
    
    #print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

    # This may take about 2 minutes to run
    # print("Neural network training initialized")
    # hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 60] #[i for i in range(1, 100)]
    # hidden_layers = []
    # accuracy_per_layer = []
    # plt.figure(figsize=(32,32))
    # for i, n_h in enumerate(hidden_layer_sizes):
    #     plt.subplot(5, 2, i+1)

    #     hidden_layers.append(n_h)
        
    #     plt.title('Hidden Layer of size %d' % n_h)
        
    #     parameters = neural_network_model(X, Y, n_h, num_iterations = 10000)
    #     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    #     predictions = predict(parameters, X)
        
    #     accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    #     accuracy_per_layer.append(accuracy)
    #     print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    
    # plt.grid(True)
    # plt.plot(hidden_layers, accuracy_per_layer, 'bo', hidden_layers, accuracy_per_layer, 'k')
    # plt.axis([0, 100, 60, 99])
    plt.show()
if __name__ == '__main__':
    main()