import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):

    W,b,U,b_tag = params

    layer_1 = np.dot(x, W) + b
    tanh_out = np.tanh(layer_1)
    probs = ll.classifier_output(tanh_out, [U, b_tag])
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

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
    # YOU CODE HERE
    return ...

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

    params = create_classifier([20, 30, 40, 10])

    for i in range(len(params)):
        print(params[i].shape)

    def _loss_and_grad(parameters):
        global params
        for i in range(len(params)):
            params[i][:] = parameters[i]
        return loss_and_gradients(np.random.randn(20), 0, params)

    for _ in range(10):
        for i in range(len(params)):
            params[i] = np.random.randn(*params[i].shape)
            gradient_check(_loss_and_grad, params[i])
