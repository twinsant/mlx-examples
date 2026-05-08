#!/usr/bin/env python3
"""
CNN 模型 + 数据集大小实验
研究不同训练集大小对 CNN vs MLP 性能的影响

使用 MLX 框架，MNIST 数据集
"""

import time
import sys
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# 导入 MNIST 数据加载器
sys.path.insert(0, "/Users/twinsant/GitHub/mlx-examples/mnist")
import mnist as mnist_data


# ============================================================
# CNN 模型
# ============================================================
class CNN(nn.Module):
    """简单 CNN for MNIST (28x28 灰度图)"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Block 1: Conv → ReLU → Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Conv → ReLU → Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC 层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, 784) → reshape to NHWC: (N, 28, 28, 1)
        x = x.reshape(-1, 28, 28, 1)

        x = nn.relu(self.conv1(x))  # (N, 28, 28, 32)
        x = self.pool1(x)           # (N, 14, 14, 32)

        x = nn.relu(self.conv2(x))  # (N, 14, 14, 64)
        x = self.pool2(x)           # (N, 7, 7, 64)

        x = x.reshape(x.shape[0], -1)  # flatten: (N, 3136)
        x = nn.relu(self.fc1(x))       # (N, 128)
        x = self.fc2(x)                # (N, 10)
        return x


# ============================================================
# MLP 模型（对照组）
# ============================================================
class MLP(nn.Module):
    """简单 MLP（隐藏层 128 宽，与 CNN 可比较）"""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# 工具函数
# ============================================================
def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")


def batch_iterate(batch_size, X, y):
    """生成等大小的 batch（跳过最后不足 batch_size 的样本）"""
    perm = mx.array(np.random.permutation(y.size))
    n_full = (y.size // batch_size) * batch_size
    for s in range(0, n_full, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def subset_data(X, y, fraction: float, seed: int = 42):
    """从数据中随机抽取 fraction 比例的子集"""
    np.random.seed(seed)
    n = X.shape[0]
    n_subset = max(int(n * fraction), 1)
    indices = np.random.choice(n, n_subset, replace=False)
    return X[mx.array(indices)], y[mx.array(indices)]


def count_params(model) -> int:
    """计算模型参数量（递归处理嵌套参数）"""
    total = 0
    for p in model.parameters().values():
        if isinstance(p, dict):
            total += count_params_in_dict(p)
        else:
            total += p.size
    return total


def count_params_in_dict(d: dict) -> int:
    total = 0
    for v in d.values():
        if isinstance(v, dict):
            total += count_params_in_dict(v)
        else:
            total += v.size
    return total


def train_and_eval(
    model: nn.Module,
    train_X, train_y,
    test_X, test_y,
    lr: float = 0.001,
    batch_size: int = 64,
    epochs: int = 15,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    训练模型并返回 (最终准确率, 训练时间, 参数量)
    注意：不使用 mx.compile（CNN 在 compile 下与可变 batch 不兼容）
    """
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    tic = time.perf_counter()
    best_acc = 0.0

    for e in range(epochs):
        for X, y in batch_iterate(batch_size, train_X, train_y):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.state)

        # 评估
        acc = mx.mean(mx.argmax(model(test_X), axis=1) == test_y)
        mx.eval(acc)
        if acc.item() > best_acc:
            best_acc = acc.item()

    toc = time.perf_counter()
    elapsed = toc - tic

    if verbose:
        n_params = count_params(model)
        print(f"  最终准确率: {best_acc:.4f}  |  时间: {elapsed:.1f}s  |  参数: {n_params:,}")

    return best_acc, elapsed, count_params(model)


# ============================================================
# 主实验
# ============================================================
def run_experiment():
    print("=" * 72)
    print("  CNN vs MLP — 数据集大小对模型性能的影响")
    print("=" * 72)

    # 加载完整数据集
    print("\n📦 加载 MNIST 数据集...")
    train_images, train_labels, test_images, test_labels = map(
        mx.array, mnist_data.mnist()
    )
    print(f"  训练集: {train_images.shape[0]:,} 张  |  测试集: {test_images.shape[0]:,} 张")

    # 数据集比例
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    fraction_labels = ["1%", "5%", "10%", "25%", "50%", "100%"]
    # 每个配置跑多次取平均
    runs_per_config = 3
    seed_base = 42

    # 训练超参数
    cnn_lr = 0.001
    mlp_lr = 0.001
    epochs = 15
    batch_size = 64

    print(f"\n⚙️  训练配置:")
    print(f"  CNN: Adam(lr={cnn_lr}), epochs={epochs}, batch={batch_size}")
    print(f"  MLP: Adam(lr={mlp_lr}), epochs={epochs}, batch={batch_size}")
    print(f"  每个配置跑 {runs_per_config} 次取平均")

    # 存结果
    results = {
        "fractions": fractions,
        "cnn_acc": [], "cnn_std": [], "cnn_time": [], "cnn_params": 0,
        "mlp_acc": [], "mlp_std": [], "mlp_time": [], "mlp_params": 0,
    }

    print("\n" + "─" * 72)
    print(f"{'数据量':>8} | {'CNN Acc':>16} | {'MLP Acc':>16} | {'CNN时间':>8} | {'MLP时间':>8}")
    print("─" * 72)

    for frac, label in zip(fractions, fraction_labels):
        cnn_accs, cnn_times = [], []
        mlp_accs, mlp_times = [], []

        for run in range(runs_per_config):
            seed = seed_base + run

            # 子采样训练集
            sub_X, sub_y = subset_data(train_images, train_labels, frac, seed=seed)

            # CNN
            cnn_model = CNN(num_classes=10)
            acc, elapsed, n_params = train_and_eval(
                cnn_model, sub_X, sub_y,
                test_images, test_labels,
                lr=cnn_lr, batch_size=batch_size, epochs=epochs,
                verbose=False,
            )
            cnn_accs.append(acc)
            cnn_times.append(elapsed)
            if results["cnn_params"] == 0:
                results["cnn_params"] = n_params

            # MLP（相同数据子集，保证公平）
            mlp_model = MLP(hidden_dim=128)
            acc, elapsed, n_params = train_and_eval(
                mlp_model, sub_X, sub_y,
                test_images, test_labels,
                lr=mlp_lr, batch_size=batch_size, epochs=epochs,
                verbose=False,
            )
            mlp_accs.append(acc)
            mlp_times.append(elapsed)
            if results["mlp_params"] == 0:
                results["mlp_params"] = n_params

        # 汇总
        cnn_mean, cnn_std = np.mean(cnn_accs), np.std(cnn_accs)
        mlp_mean, mlp_std = np.mean(mlp_accs), np.std(mlp_accs)
        cnn_t = np.mean(cnn_times)
        mlp_t = np.mean(mlp_times)

        results["cnn_acc"].append(cnn_mean)
        results["cnn_std"].append(cnn_std)
        results["cnn_time"].append(cnn_t)
        results["mlp_acc"].append(mlp_mean)
        results["mlp_std"].append(mlp_std)
        results["mlp_time"].append(mlp_t)

        print(
            f"{label:>8} | {cnn_mean:.4f} ± {cnn_std:.4f}"
            f"  | {mlp_mean:.4f} ± {mlp_std:.4f}"
            f"  | {cnn_t:>6.1f}s | {mlp_t:>6.1f}s"
        )

    print("─" * 72)
    print(f"\n📊 参数量对比: CNN={results['cnn_params']:,}  vs  MLP={results['mlp_params']:,}")

    # ============================================================
    # 分析输出
    # ============================================================
    print("\n" + "=" * 72)
    print("  📈 关键发现")
    print("=" * 72)

    # CNN vs MLP 在不同数据量下的差距
    for i, label in enumerate(fraction_labels):
        gap = results["cnn_acc"][i] - results["mlp_acc"][i]
        direction = "CNN > MLP" if gap > 0 else "MLP > CNN"
        print(f"  {label:>5}: CNN {results['cnn_acc'][i]:.4f}  MLP {results['mlp_acc'][i]:.4f}  "
              f"差距 {gap:+.4f} ({direction})")

    # 样本效率：CNN 用多少数据能达到 MLP 用全量数据的水平
    mlp_full_acc = results["mlp_acc"][-1]
    print(f"\n  🎯 MLP 全量数据准确率: {mlp_full_acc:.4f}")
    for i, label in enumerate(fraction_labels):
        cnn_acc = results["cnn_acc"][i]
        if cnn_acc >= mlp_full_acc:
            print(f"     CNN 在 {label} 数据时就达到了 MLP 全量数据的水平!")
            break
    else:
        cnn_full = results["cnn_acc"][-1]
        print(f"     CNN 全量: {cnn_full:.4f}, MLP 全量: {mlp_full_acc:.4f}")

    # 小样本下的表现差距
    print(f"\n  📉 小样本 (1%) 下 CNN 相对 MLP 的优势: "
          f"{(results['cnn_acc'][0] - results['mlp_acc'][0]) * 100:.1f} 个百分点")

    print("\n✅ 实验完成！")


if __name__ == "__main__":
    run_experiment()
