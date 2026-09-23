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

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task."""
    
    rng = np.random.RandomState(seed)
    
    # ---- 1. Generate a nonlinearly separable dataset ----
    points_per_class = 100
    N_total = points_per_class * num_classes
    
    if input_dim >= 2:
        X_core = np.zeros((N_total, 2))
        y = np.zeros(N_total, dtype=int)
        for k in range(num_classes):
            ix = range(points_per_class * k, points_per_class * (k + 1))
            r = np.linspace(0.05, 1.0, points_per_class)
            t = np.linspace(k * 4, (k + 1) * 4, points_per_class) + rng.randn(points_per_class) * 0.2
            X_core[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = k
        if input_dim > 2:
            pad = rng.randn(N_total, input_dim - 2) * 0.05
            X = np.hstack([X_core, pad])
        else:
            X = X_core
    else:
        x_line = rng.uniform(-1.0, 1.0, size=N_total)
        freq = 3
        bins = np.floor((x_line + 1.0) / 2.0 * freq * num_classes).astype(int) % num_classes
        y = bins
        X = x_line.reshape(-1, 1) + rng.randn(N_total, 1) * 0.01
    
    perm = rng.permutation(N_total)
    X = X[perm]
    y = y[perm]
    
    # ---- 2. Build a network with real nonlinear capacity ----
    hidden = 100
    init = lambda i, o: initialize_weights(i, o, scheme='he')
    
    layers = [
        make_dense(input_dim, hidden, init),
        make_activation('relu'),
        make_dense(hidden, hidden, init),
        make_activation('relu'),
        make_dense(hidden, num_classes, init),
    ]
    model = make_sequential(layers)
    
    loss_fn = make_loss('cross_entropy')
    optimizer = make_optimizer(model['params'], lr=1.0, kind='sgd')
    
    # ---- 3. Train (full-batch gradient descent, deterministic) ----
    epochs = 1000
    train(model, loss_fn, optimizer, X, y, epochs=epochs, batch_size=N_total, seed=seed)
    
    # ---- 4. Evaluate ----
    logits, _ = model['forward'](X)
    preds = np.argmax(logits, axis=1)
    accuracy = float(np.mean(preds == y))
    
    metrics = {
        'accuracy': accuracy,
        'x': X,
        'y': y,
    }
    
    return model, metrics

# Step 13 - improve_generalization
import numpy as np
import copy


def _deep_copy_params(p):
    """Recursively copy a nested params structure (dict/list of ndarrays)."""
    if isinstance(p, dict):
        return {k: _deep_copy_params(v) for k, v in p.items()}
    elif isinstance(p, (list, tuple)):
        return [_deep_copy_params(v) for v in p]
    elif isinstance(p, np.ndarray):
        return p.copy()
    else:
        raise TypeError(f"Unsupported param leaf type: {type(p)}")


def _restore_params_inplace(p, snapshot):
    """Recursively overwrite p's arrays in place with values from snapshot."""
    if isinstance(p, dict):
        for k in p:
            _restore_params_inplace(p[k], snapshot[k])
    elif isinstance(p, (list, tuple)):
        for i in range(len(p)):
            _restore_params_inplace(p[i], snapshot[i])
    elif isinstance(p, np.ndarray):
        p[...] = snapshot
    else:
        raise TypeError(f"Unsupported param leaf type: {type(p)}")


def _apply_weight_decay_inplace(p, decay):
    """Recursively shrink every parameter array toward zero in place."""
    if isinstance(p, dict):
        for k in p:
            _apply_weight_decay_inplace(p[k], decay)
    elif isinstance(p, (list, tuple)):
        for i in range(len(p)):
            _apply_weight_decay_inplace(p[i], decay)
    elif isinstance(p, np.ndarray):
        p *= (1.0 - decay)
    else:
        raise TypeError(f"Unsupported param leaf type: {type(p)}")


def _accuracy(model, x, y):
    logits, _ = model['forward'](x)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y)), preds


def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline."""
    
    N_train = x_train.shape[0]
    epochs = 500
    batch_size = min(32, N_train)
    lr = 0.5
    weight_decay = 1e-3
    
    # ---- Baseline: identical init, plain unregularized SGD, fixed epochs ----
    np.random.seed(seed)
    baseline_model = baseline_model_fn()
    baseline_loss_fn = make_loss('cross_entropy')
    baseline_optimizer = make_optimizer(baseline_model['params'], lr=lr, kind='sgd')
    train(baseline_model, baseline_loss_fn, baseline_optimizer,
          x_train, y_train, epochs=epochs, batch_size=batch_size, seed=seed)
    baseline_val_accuracy, _ = _accuracy(baseline_model, x_val, y_val)
    
    # ---- Improved: SAME initialization, but early stopping + weight decay ----
    np.random.seed(seed)
    improved_model = baseline_model_fn()
    improved_loss_fn = make_loss('cross_entropy')
    improved_optimizer = make_optimizer(improved_model['params'], lr=lr, kind='sgd')
    
    rng = np.random.RandomState(seed)
    best_val_acc = -1.0
    best_snapshot = _deep_copy_params(improved_model['params'])
    
    for epoch in range(epochs):
        perm = rng.permutation(N_train)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]
        
        for start in range(0, N_train, batch_size):
            end = min(start + batch_size, N_train)
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            train_step(improved_model, improved_loss_fn, improved_optimizer, x_batch, y_batch)
            _apply_weight_decay_inplace(improved_model['params'], weight_decay)
        
        val_acc, _ = _accuracy(improved_model, x_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_snapshot = _deep_copy_params(improved_model['params'])
    
    # Restore the best-performing weights seen during training
    _restore_params_inplace(improved_model['params'], best_snapshot)
    
    val_accuracy, predictions = _accuracy(improved_model, x_val, y_val)
    
    return {
        'val_accuracy': val_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'predictions': predictions,
        'model': improved_model,
    }

