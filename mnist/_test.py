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
    return nn.losses.cross_entropy(model(X), y, reduction="mean")

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s:s+batch_size]
        yield X[ids], y[ids]

np.random.seed(0)
train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())
print(f'Train samples: {train_labels.size}, Test samples: {test_labels.size}')

n_total = train_labels.size
n_sample = max(10, int(n_total * 0.1))
rng = np.random.RandomState(42)
indices = rng.choice(n_total, size=n_sample, replace=False)
train_images_sub = train_images[mx.array(indices)]
train_labels_sub = train_labels[mx.array(indices)]
print(f'Subsampled: {n_sample}')

effective_batch = min(256, max(1, n_sample // 5))

model = MLP(2, train_images.shape[-1], 32, 10)
mx.eval(model.parameters())
optimizer = optim.SGD(learning_rate=0.1)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

@partial(mx.compile, inputs=model.state, outputs=model.state)
def step(X, y):
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    return loss

@partial(mx.compile, inputs=model.state)
def eval_fn(X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

start = time.perf_counter()
for e in range(10):
    for X, y in batch_iterate(effective_batch, train_images_sub, train_labels_sub):
        step(X, y)
        mx.eval(model.state)
    train_acc = eval_fn(train_images_sub, train_labels_sub)
    test_acc = eval_fn(test_images, test_labels)
    print(f'E{e} train_acc={train_acc.item():.4f} test_acc={test_acc.item():.4f}')
elapsed = time.perf_counter() - start
print(f'elapsed={elapsed:.2f}s')
