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
    
    self.conv_params = {}
    self.bn_params = {}
    self.dropout_params = {}
    
    self.net = net
    self.loss_fn = loss_fn
    self.reg = reg
    self.dtype = dtype
    self.num_hlayers = len(net)
    
    # Initialize params and hyper params
    ci, hi, wi = input_dim
    
    for i, opt in enumerate(net):
      name = opt[0]
      if name == 'conv':
        name, num_filters, filter_size, stride = opt
        if hi % stride or wi % stride:
            raise ValueError('Layer [%d] cannot take %s strides in h x w: %s x %s' % (i, stride, hi, wi))
            
        params['W' + str(i)] = np.random.randn(num_filters, ci, filter_size, filter_size) * weight_scale
        params['b' + str(i)] = np.zeros(num_filters)
        
        self.conv_params[i] = {
          'stride': stride
          'pad': filter_size // 2
        }
        ci = num_filters
        hi /= stride
        wi /= stride
        
      elif name == 'bn':
        params['gamma' + str(i)] = np.ones(ci)
        params['beta' + str(i)] = np.zeros(ci)
        self.bn_params[i] = {
          'eps': 1e-5,
          'momentum': 0.9
        }
      elif name == 'pool2x2':
        # use conv with stride 2 filter size 2 would be a better idea
        wi /= 2
        hi /= 2
      elif name == 'affine':
        name, hidden_dim = opt
        params['W' + str(i)] = np.random.randn(ci * hi * wi, hidden_dim) * weight_scale
        params['b' + str(i)] = np.zeros(hidden_dim)
        
      elif name == 'dropout':
        name, p = opt
        dropout_params[i] = {
          p: p
        }
      elif name == 'relu':
        pass
      else:
        raise ValueError('invalid layer type %s' % opt[0]) 
            
      for k, v in params_i.iteritems():
        params_i[k] = v.astype(dtype)
    
      self.params[i] = params_i
        
        
     
 
  def loss(self, X, y=None):
    mode = 'test' if y is None else 'train'
        
    X = X.astype(self.dtype)
    
    pool2x2_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    params = self.params
    
    cache = {}
    
    # Forward
    for i, opt in enumerate(self.net):
      name = opt[0]
      if name == 'conv':
        W, b = params['W' + str(i)], params['b' + str(i)]
        X, cache[i] = conv_forward_fast(X, W, b, self.conv_params[i])
      elif name == 'affine':
        W, b = params['W' + str(i)], params['b' + str(i)]
        X, cache[i] = affine_forward(X, W, b)
      elif name == 'bn':
        gamma, beta= params['gamma' + str(i)], params['beta' + str(i)]
        bn_params_i = self.bn_params[i]
        bn_params_i['mode'] = mode
        X, cache[i] = spatial_batchnorm_forward(X, gamma, beta, bn_params_i)
      elif name == 'pool2x2':
        X, cache[i] = max_pool_forward_fast(X, pool2x2_param)
      elif name == 'dropout':
        Xi, cache[i] = dropout_forward(Xi, self.dropout_params[i])
      elif name == 'relu':
        Xi, cache[i] = relu_forward(Xi)
      else:
        raise ValueError('invalid layer type %s' % opt[0]) 

    scores = Xi
    
    if y is None:
      return scores
    
    
    # Backward
    
    loss, grads = 0, {}
    reg = self.reg
    L = len(self.net)
    
    loss, dscore = softmax_loss(scores, y)
    Wss = 0
    dX = loss
    
    
    for j, opt in enumerate(self.net[:,:,-1]):
      i = L - j
      name = opt[0]
      if name == 'conv':
        W, b = params['W' + str(i)], params['b' + str(i)]
        dX, grad[W + str(i)], grad[W + str(i)] = conv_backward_fast(dX, cache[i])
        Wss += np.sum(W*W)
      elif name == 'bn':
        params_i = self.params[i]
        bn_params_i = self.bn_params[i]
        bn_params_i['mode'] = mode
        Xi, cache[i] = spatial_batchnorm_forwards(Xi, params_i['gamma'], params_i['beta'], bn_params_i)
      elif name == 'pool2x2': 
        Xi, cache[i] = max_pool_forward_fast(Xi, pool2x2_param)
      elif name == 'affine':
        params_i = self.params[i]
        Xi, cache[i] = affine_forward(Xi, params_i['W'], params_i['b'])
      elif name == 'dropout':
        dropout_params_i = self.dropout_params[i]
        Xi, cache[i] = dropout_forward(Xi, self.dropout_params_i[i])
      elif name == 'relu':
        Xi, cache[i] = conv_forward_fast(Xi)
      else:
        raise ValueError('invalid layer type %s' % opt[0]) 
        
        

    
    
    loss += .5 * reg * Wss
    
    grads['W3'] = dW3 + reg * W3
    grads['b3'] = db3


    return loss, grads
  
  
pass