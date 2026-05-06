#!/usr/bin/env python3
"""MNIST 流形可视化：PCA vs t-SNE vs UMAP"""

import pickle, gzip, os, sys
import numpy as np

# --- 加载数据 ---
data_dir = os.path.expanduser("~/GitHub/mlx-examples/mnist/data")
pkl_path = os.path.join(data_dir, "mnist.pkl")

if not os.path.exists(pkl_path):
    # 从 gzip 原始文件加载
    def load_idx(path):
        with gzip.open(path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16 if 'image' in path else 8)
    images = np.concatenate([
        load_idx(os.path.join(data_dir, "train-images-idx3-ubyte.gz")).reshape(-1, 784),
        load_idx(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")).reshape(-1, 784)
    ]) / 255.0
    labels = np.concatenate([
        load_idx(os.path.join(data_dir, "train-labels-idx1-ubyte.gz")),
        load_idx(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    ])
else:
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    images = np.concatenate([d['training_images'], d['test_images']])
    labels = np.concatenate([d['training_labels'], d['test_labels']])

# 采样 10000 张加速计算
np.random.seed(42)
idx = np.random.choice(len(images), 10000, replace=False)
X = images[idx]
y = labels[idx]

print(f"数据: {X.shape[0]} 张, {X.shape[1]} 维")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
          '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990']
cmap = ListedColormap(colors)

out_dir = os.path.expanduser("~/GitHub/mlx-examples/mnist/viz")
os.makedirs(out_dir, exist_ok=True)

def plot_2d(embedding, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=cmap,
                         s=3, alpha=0.7, vmin=0, vmax=9)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    # 图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
               markersize=8, label=str(i)) for i in range(10)]
    ax.legend(handles=handles, title='Digit', loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()
    print(f"  已保存: {filename}")

# --- 1. PCA ---
print("\n[1/3] PCA 降维到 2D...")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plot_2d(X_pca, f'PCA (explained variance: {sum(pca.explained_variance_ratio_):.1%})', 'pca_2d.png')

# --- 2. t-SNE ---
print("\n[2/3] t-SNE 降维到 2D...")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X)
plot_2d(X_tsne, 't-SNE (perplexity=30)', 'tsne_2d.png')

# --- 3. UMAP ---
print("\n[3/3] UMAP 降维到 2D...")
try:
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)
    plot_2d(X_umap, 'UMAP (n_neighbors=15)', 'umap_2d.png')
except ImportError:
    print("  umap-learn 未安装，跳过。安装命令: pip install umap-learn")

print(f"\n完成！图片保存在: {out_dir}/")
