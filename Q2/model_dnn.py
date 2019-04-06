import numpy as np
from ann_utils import *
import matplotlib.pyplot as plt
def L_layer_model(X, Y, layers_dims, batch = 100, learning_rate=0.0075, num_iterations=1000, print_cost=False):
    """
    l-layer neural network all with [LINEAR->SIGMOID]*(L)

    Arguments:
    X -- input data
    Y -- 10-class poker hands
    layers_dims -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- print cost when True.

    Returns:
    parameters -- parameters learnt by the model, to used during prediction.
    """

    np.random.seed(1)
    costs = []  # keep track of cost
    m = X.shape[1]
    # Parameters initialization.
    ### START CODE HERE ###
    parameters = init_params_dnn(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        for mini_batch in range(int(m/batch)):
            # Forward propagation: all LINEAR -> SIGMOID.
            AL, caches = dnn_model_forward(X[:,mini_batch*batch:(mini_batch+1)*batch], parameters)
            # Compute cost.
            cost = compute_cost(AL, Y[:,mini_batch*batch:(mini_batch+1)*batch])

            # Backward propagation.
            grads = dnn_model_backward(AL, Y[:,mini_batch*batch:(mini_batch+1)*batch], caches)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

    #     # Print the cost every 100 training example
    #     if print_cost and i % 100 == 0:
    #         print("Cost after iteration %i: %f" % (i, cost))
    #     if print_cost and i % 100 == 0:
    #         costs.append(cost)
    #
    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters

def adaptive_learning_dnn_model(X, Y, layers_dims, batch = 100, learning_rate=0.1, tolerance = 1e-4, num_iterations=1000,  print_cost=False):
    """
    l-layer neural network all with [LINEAR->SIGMOID]*(L)

    Arguments:
    X -- input data
    Y -- 10-class poker hands
    layers_dims -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- print cost when True.

    Returns:
    parameters -- parameters learnt by the model, to used during prediction.
    """

    np.random.seed(1)
    costs = []  # keep track of cost
    cost_current = 0
    m = X.shape[1]
    learning_rate = learning_rate
    tol = tolerance
    cost_prev =0

    # Parameters initialization.
    parameters = init_params_dnn(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        for mini_batch in range(int(m/batch)):
            # Forward propagation: all LINEAR -> SIGMOID.
            AL, caches = dnn_model_forward(X[:,mini_batch*batch:(mini_batch+1)*batch], parameters)
            # Compute cost.
            cost = compute_cost(AL, Y[:,mini_batch*batch:(mini_batch+1)*batch])
            cost_current = cost
            # Backward propagation.
            grads = dnn_model_backward(AL, Y[:,mini_batch*batch:(mini_batch+1)*batch], caches)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

        # # Print the cost every 100 training example
        # if print_cost and i % 100 == 0:
        #     print("Cost after iteration %i: %f" % (i, cost))
        #     costs.append(cost)

        if (cost_current - cost_prev) < tol:
            learning_rate = learning_rate/5

    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters

def adaptive_learning_dnn_model_with_relu(X, Y, layers_dims, batch = 100, learning_rate=0.1, tolerance = 1e-4, num_iterations=1000,  print_cost=False):
    """
    l-layer neural network all with [LINEAR->ReLu]*(L-1) [LINEAR->SIGMOID]*(L)

    Arguments:
    X -- input data
    Y -- 10-class poker hands
    layers_dims -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- print cost when True.

    Returns:
    parameters -- parameters learnt by the model, to used during prediction.
    """

    np.random.seed(1)
    costs = []  # keep track of cost
    cost_current = 0
    m = X.shape[1]
    learning_rate = learning_rate
    tol = tolerance
    cost_prev =0

    # Parameters initialization.
    parameters = init_params_dnn(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        for mini_batch in range(int(m/batch)):
            # Forward propagation: all LINEAR -> SIGMOID.
            AL, caches = dnn_model_forward_with_relu(X[:,mini_batch*batch:(mini_batch+1)*batch], parameters)

            # Compute cost.
            cost = compute_cost(AL, Y[:,mini_batch*batch:(mini_batch+1)*batch])
            cost_current = cost

            # Backward propagation.
            grads = dnn_model_backward_with_relu(AL, Y[:,mini_batch*batch:(mini_batch+1)*batch], caches)

            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

        # # Print the cost every 100 training example
        # if print_cost and i % 100 == 0:
        #     print("Cost after iteration %i: %f" % (i, cost))
        #     costs.append(cost)

        if (cost_current - cost_prev) < tol:
            learning_rate = learning_rate/5

    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters