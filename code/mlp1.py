import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    W,b,U,b_tag = params

    layer_1 = np.dot(x, W) + b
    tanh_out = np.tanh(layer_1)
    probs = ll.classifier_output(tanh_out, [U, b_tag])
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W,b,U,b_tag = params
    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    # Backpropagation
    y_hat[y] -= 1
    layer_1 = np.dot(x, W) + b
    tanh_out = np.tanh(layer_1)
    gU = np.outer(tanh_out, y_hat)
    gb_tag = y_hat
    y_hat = np.dot(y_hat, U.T) * (1 - tanh_out**2)
    gW = np.outer(x, y_hat)
    gb = y_hat

    return loss,[gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.randn(in_dim, hid_dim)
    b = np.random.randn(hid_dim)
    U = np.random.randn(hid_dim, out_dim)
    b_tag = np.random.randn(out_dim)
    params = [W, b, U, b_tag]
    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3,4,5)
    params = [W, b, U, b_tag]


    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[1]

    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W, b, U
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[3]


    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0],U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)