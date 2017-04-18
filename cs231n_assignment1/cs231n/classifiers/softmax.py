import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]
    num_features = X.shape[1]
    num_classes = W.shape[1]

    for i in xrange(num_train):
        f = X[i].dot(W)
        # add stability log C = - max(f)
        f -= np.max(f)

        # calculate probability
        p = np.exp(f) / np.sum(np.exp(f))
        p_correct = p[y[i]]

        loss -= np.log(p_correct)

        dW[:, y[i]] -= X[i]

        for j in xrange(num_classes):
            dW[:, j] += p[j] * X[i]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W) / 2
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]

    f = X.dot(W)
    f -= np.max(f, axis=1)[:, np.newaxis]
    f_exp = np.exp(f)

    p = f_exp / np.sum(f_exp, axis=1)[:, np.newaxis]
    p_correct = p[np.arange(f.shape[0]), y]

    loss = - np.log(p_correct).sum() / num_train + reg * np.sum(W * W) / 2

    p[np.arange(f.shape[0]), y] -= 1

    dW = X.T.dot(p) / num_train + reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
