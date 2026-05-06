source /Users/twinsant/.openclaw/workspace/mlx-env/bin/activate
cd /Users/twinsant/GitHub/mlx-examples/mnist

python3 -c "
import time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import mnist

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)

def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction='mean')

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s:s+batch_size]
        yield X[ids], y[ids]

def train_experiment(name, num_layers, hidden_dim, lr, batch_size, epochs, optim_type='sgd'):
    np.random.seed(0)
    train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())
    
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, 10)
    mx.eval(model.parameters())
    
    if optim_type == 'adam':
        optimizer = optim.Adam(learning_rate=lr)
    else:
        optimizer = optim.SGD(learning_rate=lr)
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    @partial(mx.compile, inputs=model.state)
    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

    last_acc = 0
    for e in range(epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            step(X, y)
            mx.eval(model.state)
        acc = float(eval_fn(test_images, test_labels))
        toc = time.perf_counter()
        last_acc = acc
        if e == epochs - 1:
            print(f'{name:<18} layers={num_layers} hidden={hidden_dim:<4} lr={lr:<8.4f} batch={batch_size:<5} optim={optim_type:<4} | final_acc={acc:.4f} time={toc-tic:.2f}s')

experiments = [
    ('baseline',        2, 32,  0.1,    256, 10, 'sgd'),
    ('lr_low',          2, 32,  0.01,   256, 10, 'sgd'),
    ('lr_high',         2, 32,  0.5,    256, 10, 'sgd'),
    ('hidden_64',       2, 64,  0.1,    256, 10, 'sgd'),
    ('hidden_128',      2, 128, 0.1,    256, 10, 'sgd'),
    ('layers_1',        1, 32,  0.1,    256, 10, 'sgd'),
    ('layers_4',        4, 32,  0.1,    256, 10, 'sgd'),
    ('batch_64',        2, 32,  0.1,    64,  10, 'sgd'),
    ('batch_1024',      2, 32,  0.1,    1024,10, 'sgd'),
    ('adam_lr_3e-4',    2, 32,  0.0003, 256, 10, 'adam'),
]

print('=' * 80)
print('MNIST 超参数实验结果')
print('=' * 80)
for name, layers, hidden, lr, batch, epochs, optim_t in experiments:
    train_experiment(name, layers, hidden, lr, batch, epochs, optim_t)
print('=' * 80)
"
