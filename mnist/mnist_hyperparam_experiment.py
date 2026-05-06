#!/usr/bin/env python3
"""MNIST 超参数实验：对比不同设置对训练效果的影响"""

import subprocess
import sys
import os

# 切到 mnist 目录
os.chdir(os.path.expanduser("~/GitHub/mlx-examples/mnist"))

# 定义实验组
experiments = [
    # 基线
    {"name": "baseline",     "layers": 2, "hidden": 32,  "lr": 0.1,   "batch": 256, "epochs": 10, "optim": "sgd"},
    # 学习率对比
    {"name": "lr_low",       "layers": 2, "hidden": 32,  "lr": 0.01,  "batch": 256, "epochs": 10, "optim": "sgd"},
    {"name": "lr_high",      "layers": 2, "hidden": 32,  "lr": 0.5,   "batch": 256, "epochs": 10, "optim": "sgd"},
    # 隐藏层宽度
    {"name": "hidden_64",    "layers": 2, "hidden": 64,  "lr": 0.1,   "batch": 256, "epochs": 10, "optim": "sgd"},
    {"name": "hidden_128",   "layers": 2, "hidden": 128, "lr": 0.1,   "batch": 256, "epochs": 10, "optim": "sgd"},
    # 网络深度
    {"name": "layers_1",     "layers": 1, "hidden": 32,  "lr": 0.1,   "batch": 256, "epochs": 10, "optim": "sgd"},
    {"name": "layers_4",     "layers": 4, "hidden": 32,  "lr": 0.1,   "batch": 256, "epochs": 10, "optim": "sgd"},
    # batch size
    {"name": "batch_64",     "layers": 2, "hidden": 32,  "lr": 0.1,   "batch": 64,  "epochs": 10, "optim": "sgd"},
    {"name": "batch_1024",   "layers": 2, "hidden": 32,  "lr": 0.1,   "batch": 1024, "epochs": 10, "optim": "sgd"},
    # Adam 优化器
    {"name": "adam_lr_3e-4", "layers": 2, "hidden": 32,  "lr": 0.0003, "batch": 256, "epochs": 10, "optim": "adam"},
]

# 写一个可配置的训练脚本
train_script = '''
import argparse, time, sys
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

p = argparse.ArgumentParser()
p.add_argument("--layers", type=int, default=2)
p.add_argument("--hidden", type=int, default=32)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--optim", type=str, default="sgd")
args = p.parse_args()

np.random.seed(0)
train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())

model = MLP(args.layers, train_images.shape[-1], args.hidden, 10)
mx.eval(model.parameters())

if args.optim == "adam":
    optimizer = optim.Adam(learning_rate=args.lr)
else:
    optimizer = optim.SGD(learning_rate=args.lr)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

@partial(mx.compile, inputs=model.state, outputs=model.state)
def step(X, y):
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    return loss

@partial(mx.compile, inputs=model.state)
def eval_fn(X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

results = []
for e in range(args.epochs):
    tic = time.perf_counter()
    for X, y in batch_iterate(args.batch, train_images, train_labels):
        step(X, y)
        mx.eval(model.state)
    acc = eval_fn(test_images, test_labels)
    toc = time.perf_counter()
    results.append(f"Epoch {e}: acc={acc.item():.4f} time={toc-tic:.2f}s")

print("\\n".join(results))
'''

# 写临时脚本
with open("_hyperparam_train.py", "w") as f:
    f.write(train_script)

print("=" * 70)
print(f"{'实验':<18} {'层数':>4} {'宽度':>4} {'学习率':>8} {'batch':>6} {'优化器':>5} | 最终acc")
print("=" * 70)

for exp in experiments:
    cmd = [
        sys.executable, "_hyperparam_train.py",
        "--layers", str(exp["layers"]),
        "--hidden", str(exp["hidden"]),
        "--lr", str(exp["lr"]),
        "--batch", str(exp["batch"]),
        "--epochs", str(exp["epochs"]),
        "--optim", exp["optim"],
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        lines = result.stdout.strip().split("\n")
        last_line = lines[-1] if lines else "N/A"
        # 提取最终 accuracy
        acc = "N/A"
        for line in lines:
            if "acc=" in line:
                acc = line.split("acc=")[1].split(" ")[0]
        print(f"{exp['name']:<18} {exp['layers']:>4} {exp['hidden']:>4} {exp['lr']:>8.4f} {exp['batch']:>6} {exp['optim']:>5} | {acc}")
    except Exception as e:
        print(f"{exp['name']:<18} FAILED: {e}")

print("=" * 70)

# 清理
os.remove("_hyperparam_train.py")
