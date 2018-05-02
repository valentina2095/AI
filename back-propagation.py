import numpy as np
from random import random
import matplotlib.pyplot as plt
# from sklearn.preprocessing import normalize

patterns = np.array([[0,0, 0],
                     [0,1, 1],
                     [1,0, 1],
                     [1,1, 0]])

# Parámetros a modificar
M = [2, 3, 1]
eta = 0.5

def network_init(M):
    network = []
    for l in range(len(M)-1):
        layer = M[l]; next_layer = M[l+1]
        w = np.array([np.array([random() for i in range(next_layer)])
                                         for j in range(layer+1)])
        network.append(w)
    return network

def transfer(x, activ_fun):
    if activ_fun == "tanh":
        return np.tanh(x)
    elif activ_fun == "relu":
        return np.maximum(x, 0)
    elif activ_fun == "linear":
        return x
    elif activ_fun == "arctan":
        return 1/np.subtract(x,1)
    else:
        return 1/(1 + np.exp(-x))

def transfer_derivative(x, activ_fun):
    if activ_fun == "tanh":
        return np.subtract(1, np.power(np.tanh(x), 2))
    elif activ_fun == "relu":
        return np.greater_equal(x, 0)*1
    elif activ_fun == "linear":
        return 1
    elif activ_fun == "arctan":
        return 1/(np.power(x, 2)+1)
    else:
        f = transfer(x, "sigmoid")
        return f*(np.subtract(1, f))

def forward_propagate(weights, inputs, activ_fun):
    inputs = np.insert(inputs, inputs.shape[1], 1, axis=1)
    inputs = np.matmul(inputs, weights)                     # activation
    layer_output = transfer(inputs, activ_fun)              # transfer function
    derivative = transfer_derivative(layer_output, activ_fun)
    return layer_output, derivative

def back_propagate(network, layer, delta):
    next_layer = network[layer+1]
    errors = next_layer * delta
    return errors

def train_network(M, patterns, eta, activ_fun="sigmoid"):
    ''' This function takes a M vector with the MLP architecture, a patterns
    matrix each row being a pattern composed by inputs on initial columns and
    desired outputs in the last columns. activ_fun is an optional parameter
    where the user can define the transfer function for the network or for
    each layer by passing a vector, sigmoid is picked by default. Transfer
    functions are ["sigmoid", "tanh", "relu", "linear", "arctan"]. '''

    # Preprocessing
    patterns = np.array(patterns)
    rng = np.max(patterns, axis=0) - np.min(patterns, axis=0)
    patterns= 1 - ((np.max(patterns, axis=0)-patterns) / rng)
    np.random.shuffle(patterns)
    inputs = patterns[:, :M[0]]
    outputs = patterns[:, -M[-1]:]
    if isinstance(activ_fun, str):
        activ_fun = [activ_fun] * (len(M)-1)

    network = network_init(M)
    derivatives = []
    change = []

    sum_errors = []
    epoch = 0
    prev_error = 0
    sum_error = np.inf
    while abs(prev_error-sum_error) > 0.001 and epoch < 1000:
        prev_error = sum_error
        sum_error = 0

        # Forward propagation
        next_inputs = inputs
        layer_outputs = [next_inputs]
        for layer, f in zip(network, activ_fun):
            next_inputs, derivative = forward_propagate(layer, next_inputs, f)
            layer_outputs.append(next_inputs)
            derivatives.append(derivative)

        # Back propagation
        output_error = np.matmul(derivative.transpose(),(outputs - inputs))
        change.append(output_error)

        for layer in reversed(range(len(network)-1)):
            errors = back_propagate(network, layer, change[0])
            layer_change = np.matmul(derivatives[layer].transpose(), errors)
            change.insert(0, layer_change)

        # Adjust network weights
        for l in range(len(network)):
            network[l] = network[l] + eta * change[l+1] * layer_outputs[l]

        sum_errors.append(sum_error)
        epoch += 1
    plt.plot(sum_errors)
    plt.show()

def test(network, patterns, activ_fun="sigmoid"):
    # preprocessing
    for layer, f in zip(network, activ_fun):
        inputs = forward_propagate(layer, inputs, f)
    error = inputs-outputs

MLP = train_network(M, patterns, eta)
