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

np.random.seed(0)
train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())

model = MLP(2, train_images.shape[-1], 32, 10)
mx.eval(model.parameters())

optimizer = optim.Adam(learning_rate=0.0003)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def step(X, y):
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    return loss

def eval_fn(X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

for e in range(10):
    tic = time.perf_counter()
    for X, y in batch_iterate(256, train_images, train_labels):
        loss = step(X, y)
        mx.eval(loss)
    acc = float(eval_fn(test_images, test_labels))
    mx.eval(acc)
    toc = time.perf_counter()
    print(f'Epoch {e}: acc={acc:.4f} time={toc-tic:.2f}s')
"
