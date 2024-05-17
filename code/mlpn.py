import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    """
    Compute the output of the classifier.

    x: input data, a single vector of shape (input_dim)
    params: a list of the form [W1, b1, W2, b2, ...]

    returns:
        probs: a vector of shape (output_dim)
    """
    activation = x
    num_tanh_layers = len(params) // 2 - 1
    for i in range(num_tanh_layers):
        W, b = params[2*i], params[2*i+1]
        activation = np.tanh(np.dot(activation, W) + b)
    probs = ll.classifier_output(activation, params[-2:])
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))


def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # Forward pass
    activations = [x]
    pre_activations = []
    num_tanh_layers = len(params) // 2 - 1

    for i in range(num_tanh_layers):
        W, b = params[2 * i], params[2 * i + 1]
        z = np.dot(activations[-1], W) + b
        pre_activations.append(z)
        activations.append(np.tanh(z))

    probs = ll.classifier_output(activations[-1], params[-2:])
    loss = -np.log(probs[y])

    # Backward pass
    grads = []
    y_hat = probs.copy()
    y_hat[y] -= 1

    # Gradient for the last layer
    gW = np.outer(activations[-1], y_hat)
    gb = y_hat
    grads = [gW, gb]

    delta = y_hat

    # Gradients for the hidden layers
    for i in range(num_tanh_layers - 1, -1, -1):
        W = params[2 * i]
        z = pre_activations[i]
        delta = np.dot(delta, params[2 * (i + 1)].T) * tanh_derivative(z)
        gW = np.outer(activations[i], delta)
        gb = delta
        grads = [gW, gb] + grads

    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        W = np.random.randn(dims[i], dims[i + 1])
        b = np.random.randn(dims[i + 1])
        params.append(W)
        params.append(b)
    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    params = create_classifier([3, 20, 30, 40, 10])

    for i in range(len(params)):
        print(params[i].shape)

    def _loss_and_grad(parameters):
        global params
        params[index] = parameters
        loss, grads = loss_and_gradients([1, 2, 3], 0, params)
        return loss, grads[index]

    for _ in range(10):
        for i in range(len(params)):
            index = i
            params[i] = np.random.random_sample(params[i].shape)
            gradient_check(_loss_and_grad, params[i])
