import numpy as np
from fast_layers import *

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  N, C, H, W = x.shape
  x_new = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
  out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  N, C, H, W = dout.shape
  dout_new = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
  dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return dx, dgamma, dbeta

def batchnorm_forward(x, gamma, beta, bn_param):
    eps = bn_param.setdefault('eps', 1e-5)
    momentum = bn_param.setdefault('momentum', 0.9)

    mode = bn_param['mode']

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        sample_mean = np.mean(x, axis=0)  # 1 * D
        sample_var = np.var(x, axis=0)  # 1 * D
        x_normalized = (x - sample_mean) / np.sqrt(sample_var+ eps)  # N * D

        out = x_normalized * gamma + beta  # N * D

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x, sample_mean, sample_var, x_normalized, gamma, eps)
    elif mode == 'test':
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_normalized * gamma + beta
        cache = None
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    x, sample_mean, sample_var, x_normalized, gamma, eps = cache

    dx_normalized = dout * gamma  # N * D
    sample_std_inv = 1 / np.sqrt(sample_var + eps)  # 1 * D
    x_mu = x - sample_mean  # N * D 
    dsample_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0) * (sample_std_inv ** 3)
    dsample_mean = -1 * (np.sum(dx_normalized * sample_std_inv, axis=0) + 2 * dsample_var * np.mean(x_mu, axis=0))
    dx = dx_normalized * sample_std_inv + 2 * dsample_var * x_mu / x.shape[0] + dsample_mean / x.shape[0]
    dgamma = np.sum(dout * x_normalized, axis=0)  # 1 * D
    dbeta = np.sum(dout, axis=0)  # 1 * D

    return dx, dgamma, dbeta

def conv_relu_forward_fast(x, w, b, conv_params):
    """
    input:
        x: (N, C, H, W)
        w: (D, C, HH, WW)
        b: (D, )
        conv_params:
            {
            stride: a value 
            pad: a value
            }
    
    return:
        out: (N, D, H', W')
        cache: (cache_conv, cache_relu)
    """
    out, cache_conv = conv_forward_fast(x, w, b, conv_params)
    out, cache_relu = relu_forward(out)

    cache = (cache_conv, cache_relu)
    return out, cache

def conv_relu_backward_fast(dout, cache):
    """
    input:
        dout: (N, D, H', W')
        cache: (cache_conv, cache_relu)
    
    return:
        dx: (N, C, H, W)
        dw: (D, C, HH, WW)
        db: (D, )
    """
    cache_conv, cache_relu = cache
    
    dx = relu_backward(dout, cache_relu)
    dx, dw, db = conv_backward_fast(dx, cache_conv)
    return dx, dw, db

def conv_relu_pool_forward_fast(x, w, b, conv_params, pool_params):
    """
    input:
        x: (N, C, H, W)
        w: (D, C, HH, WW)
        b: (D, )
        conv_params: a dict
        pool_params: a dict
    
    return:
        out: (N, D, H', W')
        cache: (cache_conv, cache_relu, cache_pool)
    """
    out, cache_conv = conv_forward_fast(x, w, b, conv_params)
    out, cache_relu = relu_forward(out)
    out, cache_pool = max_pool_forward_fast(out, pool_params)

    cache = (cache_conv, cache_relu, cache_pool)
    return out, cache

def conv_relu_pool_backward_fast(dout, cache):
    """
    input:
        dout: (N, D, H', W')
        cache: (cache_conv, cache_relu, cache_pool)
    
    return:
        dx: (N, C, H, W)
        dw: (D, C, H, W)
        db: (D, )
    """
    cache_conv, cache_relu, cache_pool = cache
    
    dx = max_pool_backward_fast(dout, cache_pool)
    dx = relu_backward(dx, cache_relu)
    dx, dw, db = conv_backward_fast(dx, cache_conv)
    return dx, dw, db

def conv_relu_pool_forward_naive(x, w, b, conv_params, pool_params):
    """
    input:
        x: (N, C, H, W)
        w: (D, C, HH, WW)
        b: (D, )
        conv_params: a dict
        pool_params: a dict
    
    return:
        out: (N, D, H', W')
        cache: (cache_conv, cache_relu, cache_pool)
    """
    out, cache_conv = conv_forward_naive(x, w, b, conv_params)
    out, cache_relu = relu_forward(out)
    out, cache_pool = max_pool_forward_naive(out, pool_params)

    cache = (cache_conv, cache_relu, cache_pool)
    return out, cache

def affline_relu_forward(x, w, b):
    """
    input:
        x: (N, D) or (N, C, H, W)
        w: (D, C)
        b: (C, )
    
    return:
        out: (N * C) or (N, C, H, W)
        cache: (cache_fc, cache_relu)
    """
    out, cache_fc = affline_forward(x, w, b)
    out, cache_relu = relu_forward(out)

    cache = (cache_fc, cache_relu)
    return out, cache

def affline_relu_backward(dout, cache):
    """
    input:
        dout: (N, C) or (N, C, H, W)
        cache: (cache_fc, cache_relu)
    return:
        dx: (N, D) or (N, C, H, W)
        dw: (D, C)
        db: (C, )  
    """
    cache_fc, cache_relu = cache

    dx = relu_backward(dout, cache_relu)
    dx, dw, db = affline_backward(dx, cache_fc)
    return dx, dw, db

def affline_forward(x, w, b):
    """
    input:
        x: (N, C, H, W)
        w: (D, M)
        b: (M, )

    return:
        out: (N, M)
        cache: (x, w, b)
    """
    x_reshape = np.reshape(x, (x.shape[0], -1))  # N * D
    out = x_reshape.dot(w) + b  # N * M
    cache = (x, w, b)
    return out, cache

def affline_backward(dout, cache):
    """
    input:
        dout: (N, M)
        cache: (x, w, b)
    
    return:
        dx: (N, C, H, W)
        dw: (D, M)
        db: (M, )
    """
    x, w, b = cache
    x_reshape = np.reshape(x, (x.shape[0], -1))  # N * D

    dx = dout.dot(w.T)  # N * D
    dw = x_reshape.T.dot(dout)  # D * M
    db = np.sum(dout, axis=0)  # 1 * M

    dx = np.reshape(dx, x.shape)
    return dx, dw, db

def relu_forward(x):
    """
    input:
        x: (N, D) or (N, D, H, W)
    
    return:
        out: (N, D) or (N, D, H, W)
        cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    input:
        dout: (N, D) or (N, D, H, W)
        cache: x
    
    return:
        dx: (N, D) or (N, D, H, W)
    """
    x = cache

    dx = dout
    dx[x <= 0] = 0
    return dx

def softmax_loss(scores, y):
    """
    input:
        scores: (N, C)
        y: (C, )
    
    return:
        loss: a Value 
        dscores: (N, C)
    """
    N, D = scores.shape

    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    pro_scroes = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = -np.sum(np.log(pro_scroes[np.arange(N), y])) / N

    dscores = pro_scroes
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return loss, dscores

def conv_relu_forward_naive(x, w, b, conv_params):
    """
    input:
        x: (N, C, H, W)
        w: (D, C, HH, WW)
        b: (D, )
        conv_params:
            {
            stride: a value 
            pad: a value
            }
    
    return:
        out: (N, D, H', W')
        cache: (cache_conv, cache_relu)
    """
    out, cache_conv = conv_forward_naive(x, w, b, conv_params)
    out, cache_relu = relu_forward(out)

    cache = (cache_conv, cache_relu)
    return out, cache

def conv_relu_backward_naive(dout, cache):
    """
    input:
        dout: (N, D, H', W')
        cache: (cache_conv, cache_relu)
    
    return:
        dx: (N, C, H, W)
        dw: (D, C, HH, WW)
        db: (D, )
    """
    cache_conv, cache_relu = cache
    
    dx = relu_backward(dout, cache_relu)
    dx, dw, db = conv_backward_naive(dx, cache_conv)
    return dx, dw, db


def conv_forward_naive(x, w, b, conv_params):
    """
    input:
        x: (N, C, H, W)
        w: (D, C, HH, WW)
        b: (D, )
        conv_params:
            {
            stride: a value 
            pad: a value
            }
        
        H' = (H - HH + 2 * pad) / stride + 1
        W' = (W - HH + 2 * pad) / stride + 1
    
    return:
        out: (N, D, H', W')
        cache: (x, w, b, conv_params)
    """
    stride = conv_params['stride']
    pad = conv_params['pad']
    N, C, H, W = x.shape
    D, C, HH, WW = w.shape

    X_pad = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    H_new = int((H - HH + 2 * pad) / stride + 1)
    W_new = int((W - WW + 2 * pad) / stride + 1)

    out = np.zeros((N, D, H_new, W_new))
    for n in range(N):
        for d in range(D):
            for hn in range(H_new):
                for wn in range(W_new):
                    hn_index = int(hn * stride)
                    wn_index = int(wn * stride)
                    window = X_pad[n, :, hn_index:hn_index + HH, wn_index:wn_index + WW]
                    out[n, d, hn, wn] = np.sum(w[d] * window) + b[d]
    
    cache = (x, w, b, conv_params)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    input:
        dout: (N, D, H', W')
        cache: (x, w, b, conv_params)
    
    return:
        dx: (N, C, H, W)
        dw: (D, C, H, W)
        db: (D, )
    """
    N, D, H_new, W_new = dout.shape
    x, w, b, conv_params = cache
    stride = conv_params['stride']
    pad = conv_params['pad']
    D, C, HH, WW = w.shape
    N, C, H, W = x.shape

    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    db = np.zeros_like(b)

    X_pad = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_pad = np.zeros_like(X_pad)

    for n in range(N):
        for d in range(D):
            for hn in range(H_new):
                for wn in range(W_new):
                    hn_index = int(hn * stride)
                    wn_index = int(wn * stride)
                    window = X_pad[n, :, hn_index:hn_index + HH, wn_index:wn_index + WW]
                    dw[d] += window * dout[n, d, hn, wn]  # += !
                    db[d] += dout[n, d, hn, wn]
                    dx_pad[n, :, hn_index:hn_index + HH, wn_index:wn_index + WW]  += w[d] * dout[n, d, hn, wn]
                
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db

def conv_relu_pool_backward_naive(dout, cache):
    """
    input:
        dout: (N, D, H', W')
        cache: (cache_conv, cache_relu, cache_pool)
    
    return:
        dx: (N, C, H, W)
        dw: (D, C, H, W)
        db: (D, )
    """
    cache_conv, cache_relu, cache_pool = cache

    dx = max_pool_backward_naive(dout, cache_pool)
    dx = relu_backward(dx, cache_relu)
    dx, dw, db = conv_backward_naive(dx, cache_conv)
    return dx, dw, db

def max_pool_forward_naive(x, pool_params):
    """
    input:
        x: (N, C, H, W)
        pool_params:
            {
            pool_width: a value 
            pool_height: a value 
            stride: a value 
            }
    
    H' = (H - pool_height) / stride + 1
    W' = (W - pool_width) / stride + 1

    return:
        out: (N, C, H', W')
        cahce: (x, pool_params)
    """
    N, C, H, W = x.shape
    pool_width = pool_params['pool_width']
    pool_height = pool_params['pool_height']
    stride = pool_params['stride']

    H_new = int((H - pool_height) / stride + 1)
    W_new = int((W - pool_width) / stride + 1)

    out = np.zeros((N, C, H_new, W_new))
    for n in range(N):
        for c in range(C):
            for hn in range(H_new):
                for wn in range(W_new):
                    hn_index = int(hn * stride)
                    wn_index = int(wn * stride)
                    window = x[n, c, hn_index:hn_index + pool_height, wn_index:wn_index + pool_width]
                    out[n, c, hn, wn] = np.max(window)
    
    cache = (x, pool_params)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    input:
        dout: (N, C, H', W')
        cache: (x, pool_params)
    
    return:
        dx: (N, C, H, W)
    """
    N, C, H_new, W_new = dout.shape
    x, pool_params = cache
    pool_width = pool_params['pool_width']
    pool_height = pool_params['pool_height']
    stride = pool_params['stride']

    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for hn in range(H_new):
                for wn in range(W_new):
                    hn_index = int(hn * stride)
                    wn_index = int(wn * stride)
                    window = x[n, c, hn_index:hn_index + pool_height, wn_index:wn_index + pool_width]
                    m = np.max(window)
                    dx[n, c, hn_index:hn_index + pool_height, wn_index:wn_index + pool_width] += (window == m) * dout[n, c, hn, wn]  # += !
    return dx


