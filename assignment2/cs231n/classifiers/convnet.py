import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


### TODO: COMPLET THIS LATER - Siyun

class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), 
               net = [
                   ('conv', 32, 7, 1),
                   ('bn'),
                   ('pool2x2'),
                   ('affine', 100),
                   ('relu'),
                   ('affine', 10)
               ],
               loss_fn = [
                   'softmax'
               ],
               weight_scale=1e-3, 
               reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.net = net
    self.loss_fn = loss_fn
    self.reg = reg
    self.dtype = dtype
    self.num_hlayers = len(net)
    
    # Initialize weights
    input_dim
    
    ci, hi, wi = input_dim
    
    for i, opt in enumerate(net):
      params_i = {}
      if opt[0] == 'conv':
        name, num_filters, filter_size, stride = opt
        params_i['W'] = np.random.randn(num_filters, ci, filter_size, filter_size) * weight_scale
        params_i['b'] = np.zeros(num_filters)]
        ci = num_filters
        hi /= stride
        wi /= stride
        
        
      elif opt[0] == 'bn':
        params_i['gamma'] = np.ones(ci)
        params_i['beta'] = np.zeros(ci)
      elif opt[0] == 'pool2x2':
        wi /= 2
        hi /= 2
      elif opt[0] == 'affine':
        name, hidden_dim = opt
        params_i['W'] = np.random.randn(ci * hi * wi, hidden_dim) * weight_scale
        params_i['b'] = np.zeros(hidden_dim)
        
      # Following does not make changes to size or parameter:
      # relu, dropout
      else:
        raise ValueError('invalid layer type %s' % opt[0]) 
            
      for k, v in params_i.iteritems():
        params_i[k] = v.astype(dtype)
    
      self.params[i] = params_i
     
 
  def loss(self, X, y=None):
    pool2x2_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    conv_param = {}
    
    # Forward
    
    
    
    
    
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    

    scores = None

    # Forward
    
    Xi, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    Xi, cache2 = affine_relu_forward(Xi, W2, b2)
    scores, cache3 = affine_forward(Xi, W3, b3)
    

    
    if y is None:
      return scores
    
    
    # Backward
    
    loss, grads = 0, {}

    
    reg = self.reg
    
    loss, dscore = softmax_loss(scores, y)
    loss += .5 * reg * (np.sum(W3*W3) + np.sum(W2*W2) + np.sum(W1*W1))
    
    dXi, dW3, db3 = affine_backward(dscore, cache3)
    dXi, dW2, db2 = affine_relu_backward(dXi, cache2)
    dX, dW1, db1 = conv_relu_pool_backward(dXi, cache1)
    
    grads['W3'] = dW3 + reg * W3
    grads['b3'] = db3
    grads['W2'] = dW2 + reg * W2
    grads['b2'] = db2
    grads['W1'] = dW1 + reg * W1
    grads['b1'] = db1

    return loss, grads
  
  
pass