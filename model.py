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

# Step 5 - initialize_weights
import numpy as np

def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer."""
    
    if scheme == 'he':
        std = np.sqrt(2.0 / in_dim)
    elif scheme == 'xavier':
        std = np.sqrt(1.0 / in_dim)
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")
    
    W = np.random.randn(in_dim, out_dim) * std
    b = np.zeros(out_dim)
    
    return W, b

# Step 6 - make_loss
import numpy as np

def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits)."""
    
    if kind != 'cross_entropy':
        raise ValueError(f"Unsupported loss kind: {kind}")
    
    def loss_fn(logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        batch_size = logits.shape[0]
        
        # Numerically stable softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        
        # Cross-entropy loss: -log(p_correct), averaged over the batch
        correct_class_probs = probs[np.arange(batch_size), labels]
        # small epsilon guards log(0) in fully-saturated edge cases
        log_probs = np.log(np.clip(correct_class_probs, 1e-12, None))
        loss = float(-np.mean(log_probs))
        
        # Gradient: (p - one_hot) / batch_size
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), labels] -= 1
        d_logits /= batch_size
        
        return loss, d_logits
    
    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model."""
    
    def forward(x):
        caches = []
        h = x
        for layer in layers:
            h, cache = layer['forward'](h)
            caches.append(cache)
        y = h
        return y, caches
    
    def backward(dout, caches):
        grads_list = [None] * len(layers)
        dh = dout
        
        # walk layers in REVERSE order
        for i in reversed(range(len(layers))):
            layer = layers[i]
            cache = caches[i]
            dh, grads = layer['backward'](dh, cache)
            grads_list[i] = grads
        
        dx = dh
        return dx, grads_list
    
    params = [layer['params'] for layer in layers]
    
    return {
        'forward': forward,
        'backward': backward,
        'params': params
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch."""
    
    # Forward pass through the model: x -> logits
    logits, caches = model['forward'](x)
    
    # Compute the scalar loss and the gradient w.r.t. the logits
    loss, d_logits = loss_fn(logits, y)
    
    # Backward pass through the model: propagate d_logits back to every parameter
    dx, param_grads = model['backward'](d_logits, caches)
    
    return loss, param_grads

# Step 9 - make_optimizer
import numpy as np

def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place."""
    
    if kind != 'sgd':
        raise ValueError(f"Unsupported optimizer kind: {kind}")
    
    def _apply_update(p, g):
        """Recursively walk matching structures of params and grads,
        updating every ndarray leaf in place."""
        if isinstance(p, dict):
            for key in p:
                _apply_update(p[key], g[key])
        elif isinstance(p, (list, tuple)):
            for i in range(len(p)):
                _apply_update(p[i], g[i])
        elif isinstance(p, np.ndarray):
            p -= lr * g
        else:
            raise TypeError(f"Unsupported parameter leaf type: {type(p)}")
    
    def step(grads):
        _apply_update(params, grads)
    
    return {'step': step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch."""
    
    # Evaluate loss and gradients on the CURRENT (pre-update) parameters
    loss, param_grads = forward_backward(model, loss_fn, x_batch, y_batch)
    
    # Apply one optimizer update using those gradients
    optimizer['step'](param_grads)
    
    # Return the loss as it was BEFORE the update above
    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop."""
    
    N = x.shape[0]
    rng = np.random.RandomState(seed)
    
    history = []
    
    for epoch in range(epochs):
        # Shuffle indices deterministically for this epoch
        perm = rng.permutation(N)
        x_shuffled = x[perm]
        y_shuffled = y[perm]
        
        batch_losses = []
        
        # Walk through the shuffled data in chunks of batch_size
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)
            batch_losses.append(loss)
        
        epoch_loss = float(np.mean(batch_losses))
        history.append(epoch_loss)
    
    return history

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

