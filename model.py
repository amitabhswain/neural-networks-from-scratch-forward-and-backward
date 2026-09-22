"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index', 'zerosize_ok'])
    while not it.finished:
        idx = it.multi_index
        
        original_value = x[idx]
        
        x[idx] = original_value + eps
        f_plus = f(x)
        
        x[idx] = original_value - eps
        f_minus = f(x)
        
        x[idx] = original_value
        
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        
        it.iternext()
    
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    a = np.asarray(analytic_grad, dtype=float)
    n = np.asarray(numeric_grad, dtype=float)
    
    numerator = np.abs(a - n)
    denominator = np.maximum(np.maximum(np.abs(a), np.abs(n)), tol)
    
    relative_error = numerator / denominator
    
    return float(np.max(relative_error))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer."""
    W, b = weight_init_fn(in_dim, out_dim)
    
    params = {'W': W, 'b': b}
    
    def forward(x):
        y = x @ params['W'] + params['b']
        cache = x
        return y, cache
    
    def backward(dout, cache):
        x = cache
        
        dx = dout @ params['W'].T
        dW = x.T @ dout
        db = np.sum(dout, axis=0)
        
        grads = {'W': dW, 'b': db}
        return dx, grads
    
    return {
        'params': params,
        'forward': forward,
        'backward': backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer."""
    
    params = {}
    
    if kind == 'relu':
        def forward(x):
            y = np.maximum(x, 0)
            cache = x
            return y, cache
        
        def backward(dout, cache):
            x = cache
            dx = dout * (x > 0)
            return dx, {}
    
    elif kind == 'tanh':
        def forward(x):
            y = np.tanh(x)
            cache = y
            return y, cache
        
        def backward(dout, cache):
            y = cache
            dx = dout * (1 - y ** 2)
            return dx, {}
    
    elif kind == 'sigmoid':
        def forward(x):
            y = np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )
            cache = y
            return y, cache
        
        def backward(dout, cache):
            y = cache
            dx = dout * y * (1 - y)
            return dx, {}
    
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    
    return {
        'params': params,
        'forward': forward,
        'backward': backward
    }

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

