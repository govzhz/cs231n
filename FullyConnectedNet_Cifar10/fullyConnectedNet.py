import numpy as np
from layer_utils import *

class FullyConnectedNet(object):
    def __init__(self, input_dim, hidden_dim, num_classes, weight_scale, dropout=0, reg=0.0, use_batchnorm=False):
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0

        self.params = {}
        layer_dims = [input_dim] + hidden_dim + [num_classes]
        self.num_layers = len(hidden_dim) + 1

        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = weight_scale * np.random.randn(layer_dims[i], layer_dims[i + 1])
            self.params['b' + str(i + 1)] = np.zeros(layer_dims[i + 1])
            if self.use_batchnorm and i < len(hidden_dim):
                self.params['gamma' + str(i + 1)] = np.ones(layer_dims[i + 1])
                self.params['beta' + str(i + 1)] = np.zeros(layer_dims[i + 1])
        
        if self.use_batchnorm:
            self.bn_configs = {}
            for i in range(self.num_layers - 1):
                self.bn_configs['W' + str(i + 1)] = {'mode': 'train'}
        
        if self.use_dropout:
            self.dp_param = {'mode': 'train', 'p': dropout}
            
    
    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        if self.use_dropout:
            self.dp_param['mode'] = mode
        
        if self.use_batchnorm:
            for bn in self.bn_configs:
                self.bn_configs[bn]['mode'] = mode

        caches = []
        cache_dropout = []
        out = X
        for i in range(self.num_layers - 1):
            W = self.params['W' + str(i + 1)]
            b = self.params['b' + str(i + 1)]

            if self.use_batchnorm:
                bn_param = self.bn_configs['W' + str(i + 1)]
                gamma = self.params['gamma' + str(i + 1)]
                beta = self.params['beta' + str(i + 1)]
                out, cache = affine_batchnorm_relu_forward(out, W, b, gamma, beta, bn_param)
            else:
                out, cache = affine_relu_forward(out, W, b)
            caches.append(cache)

            if self.use_dropout:
                out, cache = dropout_forward(out, self.dp_param)
                cache_dropout.append(cache)
        
        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        out, cache = affine_forward(out, W, b)
        caches.append(cache)
        scores = out

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        for i in range(self.num_layers):
            W = self.params['W' + str(i + 1)]
            loss += 0.5 * self.reg * np.sum(W * W)
        
        grads = {}
        dout, dw, db = affine_backward(dscores, caches[-1])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db
        for i in range(self.num_layers - 1)[::-1]:
            cache = caches[i]
            
            if self.use_dropout:
                dout = dropout_backward(dout, cache_dropout[i])

            if self.use_batchnorm:
                dout, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache)
                grads['gamma' + str(i + 1)] = dgamma
                grads['beta' + str(i + 1)] = dbeta
            else:
                dout, dw, db = affine_relu_backward(dout, cache)
            
            grads['W' + str(i + 1)] = dw + self.reg * self.params['W' + str(i + 1)]
            grads['b' + str(i + 1)] = db
        
        return loss, grads

