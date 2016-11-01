import numpy as np

def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    next_w = w - config['learning_rate'] * dw 
    return next_w, config

def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    # v is a variable, so we needn't to setdefault'
    v = config.get('velocity', np.zeros_like(w))

    learning_rate = config['learning_rate']
    momentum = config['momentum']

    v = momentum * v - learning_rate * dw
    next_w = w + v
    config['velocity'] = v

    return next_w, config

def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    t = config.get('t', 0)
    v = config.get('v', np.zeros_like(w))
    m = config.get('m', np.zeros_like(w))

    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']

    t += 1
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    m_bias = m / (1 - beta1 ** t)
    v_bias = v / (1 - beta2 ** t)
    next_w = w - learning_rate * m_bias / (np.sqrt(v_bias) + epsilon)

    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_w, config

def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    epsilon = config.setdefault('epsilon', 1e-8)
    cache = config.get('cache', np.zeros_like(w))

    learning_rate = config['learning_rate']
    decay_rate = config['decay_rate']
    epsilon = config['epsilon']

    cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)
    next_w = w - learning_rate * dw / (np.sqrt(cache) + epsilon)

    config['cache'] = cache
    return next_w, config