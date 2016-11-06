import numpy as np
from layer_utils import * 

class CNN(object):
    def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=3, hidden_dims=100, num_classes=10, weight_scales=1e-3, reg=0.0, use_batchnorm=False):
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.params = {}
        C, H, W = input_dims[0], input_dims[1], input_dims[1]

        self.params['W1'] = weight_scales * np.random.randn(32, 3, 3, 3)
        self.params['b1'] = np.zeros(32)
        self.params['W2'] = weight_scales * np.random.randn(64, 32, 3, 3)
        self.params['b2'] = np.zeros(64)
        self.params['W3'] = weight_scales * np.random.randn(128, 64, 3, 3)
        self.params['b3'] = np.zeros(128)
        self.params['W4'] = weight_scales * np.random.randn(256, 128, 3, 3)
        self.params['b4'] = np.zeros(256)
        self.params['W5'] = weight_scales * np.random.randn(int(8 * 8 * 256), 384)
        self.params['b5'] = np.zeros(384)
        self.params['W6'] = weight_scales * np.random.randn(384, 10)
        self.params['b6'] = np.zeros(10)
        
        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(64)
            self.params['beta1'] = np.zeros(64)
            self.params['gamma2'] = np.ones(256)
            self.params['beta2'] = np.zeros(256)
            self.bn_param1 = {'mode': 'train'}
            self.bn_param2 = {'mode': 'train'}

    
    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        if self.use_batchnorm:
            self.bn_param1['mode'] = mode
            self.bn_param2['mode'] = mode
        
        conv_param = {'stride': 1, 'pad': int((3 - 1) / 2)}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # conv1 
        W1 = self.params['W1']
        b1 = self.params['b1']
        out, cache_conv1 = conv_relu_forward_fast(X, W1, b1, conv_param)


        # conv2 and pool(or norm)
        W2 = self.params['W2']
        b2 = self.params['b2']
        out, cache_conv2 = conv_relu_pool_forward_fast(out, W2, b2, conv_param, pool_param)
        if self.use_batchnorm:
            out, cache_batch1 = spatial_batchnorm_forward(out, self.params['gamma1'], self.params['beta1'], self.bn_param1)

        # conv3
        W3 = self.params['W3']
        b3 = self.params['b3']
        out, cache_conv3 = conv_relu_forward_fast(out, W3, b3, conv_param)

        # conv4 and pool(or norm)
        W4 = self.params['W4']
        b4 = self.params['b4']
        out, cache_conv4 = conv_relu_pool_forward_fast(out, W4, b4, conv_param, pool_param)
        if self.use_batchnorm:
            out, cache_batch2 = spatial_batchnorm_forward(out, self.params['gamma2'], self.params['beta2'], self.bn_param2)
        
        # fc
        W5 = self.params['W5']
        b5 = self.params['b5']
        out, cache_fc1 = affline_relu_forward(out, W5, b5)

        # fc(softmax)
        W6 = self.params['W6']
        b6 = self.params['b6']
        out, cache_fc2 = affline_forward(out, W6, b6)

        scores = out
        if y is None:
            return scores
        
        # loss
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * sum(np.sum(w * w) for w in [W1, W2, W3, W4, W5, W6])

        # dfc2
        dx, dw_fc2, db_fc2 = affline_backward(dscores, cache_fc2)

        # dfc1
        dx, dw_fc1, db_fc1 = affline_relu_backward(dx, cache_fc1)

        # dconv4
        if self.use_batchnorm:
            dx, dgamma2, dbeta2 = spatial_batchnorm_backward(dx, cache_batch2)
        dx, dw_conv4, db_conv4 = conv_relu_pool_backward_fast(dx, cache_conv4)

        # dconv3
        dx, dw_conv3, db_conv3 = conv_relu_backward_fast(dx, cache_conv3)

        # dconv2
        if self.use_batchnorm:
            dx, dgamma1, dbeta1 = spatial_batchnorm_backward(dx, cache_batch1)
        dx, dw_conv2, db_conv2 = conv_relu_pool_backward_fast(dx, cache_conv2)

        # dconv1
        dx, dw_conv1, db_conv1 = conv_relu_backward_fast(dx, cache_conv1)

        # grads
        grads = {}
        if self.use_batchnorm:
            grads['gamma1'] = dgamma1
            grads['beta1'] = dbeta1
            grads['gamma2'] = dgamma2
            grads['beta2'] = dbeta2

        grads['W1'] = dw_conv1 + self.reg * W1
        grads['b1'] = db_conv1
        grads['W2'] = dw_conv2 + self.reg * W2
        grads['b2'] = db_conv2
        grads['W3'] = dw_conv3 + self.reg * W3
        grads['b3'] = db_conv3
        grads['W4'] = dw_conv4 + self.reg * W4
        grads['b4'] = db_conv4
        grads['W5'] = dw_fc1 + self.reg * W5
        grads['b5'] = db_fc1
        grads['W6'] = dw_fc2 + self.reg * W6
        grads['b6'] = db_fc2
        
        return loss, grads

