import numpy as np

def affine_relu_forward(x, w, b):
    out_fc, cache_fc = affine_forward(x, w, b)
    out, cache_relu = relu_forward(out_fc)
    cache = (cache_fc, cache_relu)
    return out, cache


def affine_relu_backward(dout, cache):
    cache_fc, cache_relu = cache
    da = relu_backward(dout, cache_relu)
    dx, dw, db = affine_backward(da, cache_fc)
    return dx, dw, db


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    out_fc, cache_fc = affine_forward(x, w, b)
    out_bn, cache_bn = batchnorm_forward(out_fc, gamma, beta, bn_param)
    out, cache_relu = relu_forward(out_bn)
    cache = (cache_fc, cache_bn, cache_relu)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    cache_fc, cache_bn, cache_relu = cache
    dout = relu_backward(dout, cache_relu)
    dout, dgamma, dbeta = batchnorm_backward(dout, cache_bn)
    dx, dw, db = affine_backward(dout, cache_fc)
    return dx, dw, db, dgamma, dbeta


def affine_forward(x, w, b):
    """
    x: N * D 
    w: D * C 
    b: 1 * C
    """
    out = x.dot(w) + b 
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    dout: N * C
    """
    x, w, b = cache

    dx = dout.dot(w.T)  # N * D
    dw = x.T.dot(dout)  # D * C 
    db = np.sum(dout, axis=0)  # 1 * N 
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache

    dx = dout
    dx[x <= 0] = 0
    return dx

def softmax_loss(scores, y):
    N = scores.shape[0]

    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    pro_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = -1 * np.sum(np.log(pro_scores[np.arange(N), y])) / N

    dscores = pro_scores
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return loss, dscores

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

def dropout_forward(x, df_param):
    p = df_param['p']
    mode = df_param['mode']

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == 'test':
        out = x
        mask = None
    
    cache = (mask, df_param)
    return x, cache

def dropout_backward(dout, cache):
    mask, df_param = cache
    mode = df_param['mode']

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx
