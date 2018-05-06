import numpy as np
from random import random
import matplotlib.pyplot as plt
# from sklearn.preprocessing import normalize

def network_init(M):
    network = []
    for l in range(len(M)-1):
        layer = M[l]; next_layer = M[l+1]
        w = np.array([np.array([random() for i in range(next_layer)])
                                         for j in range(layer)])
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
    # inputs = np.insert(inputs, inputs.shape[1], 1, axis=1)  # bias
    inputs = np.matmul(inputs, weights)                     # activation
    layer_output = transfer(inputs, activ_fun)              # transfer function
    derivative = transfer_derivative(inputs, activ_fun)
    return layer_output, derivative

def back_propagate(layer, deriv, delta, layer_input):
    new_delta = np.matmul(delta, layer.T)*deriv
    dJ = np.matmul(layer_input.T, new_delta)
    return new_delta, dJ

def train_network(M, data, eta, activ_fun="sigmoid"):
    ''' This function takes a M vector with the MLP architecture, a data
    matrix each row being a pattern composed by inputs on initial columns and
    desired outputs in the last columns. activ_fun is an optional parameter
    where the user can define the transfer function for the network or for
    each layer by passing a vector, sigmoid is picked by default. Transfer
    functions are ["sigmoid", "tanh", "relu", "linear", "arctan"]. '''

    # Preprocessing
    data = np.array(data)
    shuff = list(range(len(data))); np.random.shuffle(shuff)
    data = data[shuff]

    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    norm_data = 1 - ((max_data - data) / (max_data - min_data))

    inputs = norm_data[:, :M[0]]
    outputs = norm_data[:, -M[-1]:]

    if isinstance(activ_fun, str):
        activ_fun = [activ_fun] * (len(M)-1)

    network = network_init(M)
    sum_errors = []
    epoch = 0
    prev_J = 0
    J = np.inf
    min_error = np.inf

    while abs(prev_J-J) > (0.0001*len(data)) and epoch < 1000:
        prev_J = J
        J = 0

        # Forward propagation
        next_inputs = inputs
        layer_inputs = []
        derivatives = []
        for layer, f in zip(network, activ_fun):
            layer_inputs.append(next_inputs)
            next_inputs, derivative = forward_propagate(layer, next_inputs, f)
            derivatives.append(derivative)

        # Back propagation
        J = 1/2 * np.sum(np.power((outputs - next_inputs), 2))
        delta = -np.multiply(-(outputs - next_inputs), derivative)
        deltas = [delta]
        dW = [np.matmul(layer_inputs[-1].T, delta)]

        for l in reversed(range(len(network)-1)):
            delta, dJ = back_propagate(network[l+1], derivatives[l], deltas[0], layer_inputs[l])
            deltas.insert(0, delta)
            dW.insert(0, dJ)

        # Save network with least error
        if J < min_error:
            least_error_network = network
            min_error_output = next_inputs
            min_error = J

        # Adjust network weights
        for l in range(len(network)):
            network[l] = network[l] + eta * dW[l]

        sum_errors.append(J)
        epoch += 1
    print('epoch:', epoch)
    plt.plot(sum_errors)
    plt.show()

    # Postprocessing
    final_out = []
    least_out = []
    for i in range(len(shuff)):
        pos = shuff.index(i)
        final_out.append(next_inputs[pos])
        least_out.append(min_error_output[pos])
    final_out = np.array(final_out)
    least_out = np.array(least_out)
    final_out = (final_out - 1) * (max_data[-1] - min_data[-1]) + max_data[-1]
    least_out = (least_out - 1) * (max_data[-1] - min_data[-1]) + max_data[-1]

    return final_out, network, least_out, least_error_network


# data = np.loadtxt('data.csv')
# M = [2, 5, 1]
# eta = 0.5
# MLP_out, MLP, least_err_out, least_err_MLP = train_network(M, data, eta)
