"""
计算图数据集嵌入到欧氏空间和双曲空间的失真度

失真度指标：
1. Average Distortion (AD): 平均失真度
2. Worst-case Distortion (WD): 最坏情况失真度
3. Mean Average Precision (MAP): 邻域保持质量
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from sklearn.manifold import MDS
import geoopt
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_graph_distances(edge_index, num_nodes):
    """计算图上的最短路径距离矩阵"""
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].cpu().numpy()
    # 使用Floyd-Warshall或Dijkstra计算最短路径
    dist_matrix = shortest_path(adj, directed=False, unweighted=True)
    # 将无穷大替换为最大有限距离的2倍（表示不连通）
    max_finite = dist_matrix[np.isfinite(dist_matrix)].max()
    dist_matrix[np.isinf(dist_matrix)] = max_finite * 2
    return dist_matrix


def euclidean_embedding(graph_dist, dim=128, random_state=42):
    """使用MDS将图嵌入到欧氏空间"""
    print(f"  Embedding to {dim}D Euclidean space using MDS...")
    mds = MDS(n_components=dim, dissimilarity='precomputed', 
              random_state=random_state, n_init=4, max_iter=300)
    embedding = mds.fit_transform(graph_dist)
    return embedding


def hyperbolic_embedding_poincare(graph_dist, dim=128, c=1.0, lr=0.1, epochs=500):
    """使用优化方法将图嵌入到双曲空间（Poincaré球模型）- 简化版本"""
    print(f"  Embedding to {dim}D Hyperbolic space (c={c})...")
    
    n = graph_dist.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机初始化（不使用MDS）
    print(f"  Initializing embeddings randomly...")
    embeddings = torch.randn(n, dim).float().to(device) * 0.01
    
    # 归一化到球内 (radius < 1/sqrt(c))
    max_norm = 0.9 / np.sqrt(c)
    norm = torch.norm(embeddings, dim=1, keepdim=True)
    embeddings = embeddings / (norm.max() + 1e-5) * max_norm * 0.3
    embeddings.requires_grad_(True)
    
    # 使用标准Adam优化器（不用Riemannian优化器）
    optimizer = torch.optim.Adam([embeddings], lr=lr)
    
    graph_dist_torch = torch.from_numpy(graph_dist).float().to(device)
    
    # 优化嵌入以最小化失真
    pbar = tqdm(range(epochs), desc="  Optimizing hyperbolic embedding")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # 投影到Poincaré球内（保持在流形上）
        with torch.no_grad():
            norm = torch.norm(embeddings, dim=1, keepdim=True)
            # 确保所有点在球内: ||x|| < 1/sqrt(c)
            max_allowed = max_norm
            embeddings.data = torch.where(
                norm > max_allowed,
                embeddings / norm * max_allowed * 0.99,
                embeddings
            )
        
        # 计算Poincaré距离
        # d(x,y) = arcosh(1 + 2*c*||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # [n, n, dim]
        diff_norm_sq = (diff ** 2).sum(dim=2)  # [n, n]
        
        x_norm_sq = (embeddings ** 2).sum(dim=1, keepdim=True)  # [n, 1]
        y_norm_sq = x_norm_sq.t()  # [1, n]
        
        numerator = 2 * c * diff_norm_sq
        denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq.t())
        denominator = torch.clamp(denominator, min=1e-8)
        
        # 使用稳定的arcosh计算
        arg = 1 + numerator / denominator
        arg = torch.clamp(arg, min=1.0 + 1e-7)  # arcosh定义域 >= 1
        poincare_dist = torch.log(arg + torch.sqrt(arg ** 2 - 1))
        
        # 计算失真损失（均方误差）
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        loss = ((poincare_dist[mask] - graph_dist_torch[mask]) ** 2).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return embeddings.detach().cpu().numpy()


def compute_euclidean_distances(embedding):
    """计算欧氏空间中的距离矩阵"""
    diff = embedding[:, np.newaxis, :] - embedding[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return distances


def compute_poincare_distances(embedding, c=1.0):
    """计算Poincaré球中的距离矩阵"""
    n = embedding.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            x, y = embedding[i], embedding[j]
            diff_norm_sq = np.sum((x - y) ** 2)
            x_norm_sq = np.sum(x ** 2)
            y_norm_sq = np.sum(y ** 2)
            
            numerator = 2 * c * diff_norm_sq
            denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
            denominator = max(denominator, 1e-8)
            
            dist = np.arccosh(1 + numerator / denominator)
            distances[i, j] = distances[j, i] = dist
    
    return distances


def compute_distortion_metrics(graph_dist, embed_dist):
    """计算失真度指标"""
    n = graph_dist.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    graph_dist_flat = graph_dist[mask]
    embed_dist_flat = embed_dist[mask]
    
    # 避免除零
    graph_dist_flat = np.maximum(graph_dist_flat, 1e-8)
    embed_dist_flat = np.maximum(embed_dist_flat, 1e-8)
    
    # 计算失真比率
    distortion_ratios = embed_dist_flat / graph_dist_flat
    
    # Average Distortion (AD)
    avg_distortion = np.mean(distortion_ratios)
    
    # Worst-case Distortion (WD)
    worst_distortion = np.max(distortion_ratios)
    
    # Standard deviation
    std_distortion = np.std(distortion_ratios)
    
    return {
        'avg_distortion': avg_distortion,
        'worst_distortion': worst_distortion,
        'std_distortion': std_distortion
    }


def compute_map(graph_dist, embed_dist, k=10):
    """计算Mean Average Precision - 评估k近邻保持情况"""
    n = graph_dist.shape[0]
    
    # 对每个节点找到图距离的k近邻
    graph_neighbors = np.argsort(graph_dist, axis=1)[:, 1:k+1]
    
    # 对每个节点找到嵌入距离的k近邻
    embed_neighbors = np.argsort(embed_dist, axis=1)[:, 1:k+1]
    
    # 计算每个节点的Average Precision
    aps = []
    for i in range(n):
        graph_nn = set(graph_neighbors[i])
        embed_nn = embed_neighbors[i]
        
        # 计算precision@k
        hits = 0
        precisions = []
        for j, neighbor in enumerate(embed_nn):
            if neighbor in graph_nn:
                hits += 1
                precisions.append(hits / (j + 1))
        
        if len(precisions) > 0:
            aps.append(np.mean(precisions))
        else:
            aps.append(0.0)
    
    return np.mean(aps)


def analyze_dataset(dataset_name, dim=128, c=1.0):
    """分析单个数据集"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据集
    dataset = Planetoid(root='./data', name=dataset_name)
    data = dataset[0]
    
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    
    print(f"Nodes: {num_nodes}, Edges: {num_edges}")
    
    # 计算图距离矩阵
    print("\n1. Computing graph distances...")
    graph_dist = compute_graph_distances(data.edge_index, num_nodes)
    
    # 欧氏空间嵌入
    print("\n2. Euclidean Space Embedding:")
    eucl_embedding = euclidean_embedding(graph_dist, dim=dim)
    eucl_dist = compute_euclidean_distances(eucl_embedding)
    
    eucl_metrics = compute_distortion_metrics(graph_dist, eucl_dist)
    eucl_map = compute_map(graph_dist, eucl_dist, k=10)
    
    print(f"  Average Distortion (AD):     {eucl_metrics['avg_distortion']:.4f}")
    print(f"  Worst-case Distortion (WD):  {eucl_metrics['worst_distortion']:.4f}")
    print(f"  Std Distortion:              {eucl_metrics['std_distortion']:.4f}")
    print(f"  MAP@10:                      {eucl_map:.4f}")
    
    # 双曲空间嵌入
    print("\n3. Hyperbolic Space Embedding:")
    hyp_embedding = hyperbolic_embedding_poincare(graph_dist, dim=dim, c=c, lr=0.1, epochs=500)
    hyp_dist = compute_poincare_distances(hyp_embedding, c=c)
    
    hyp_metrics = compute_distortion_metrics(graph_dist, hyp_dist)
    hyp_map = compute_map(graph_dist, hyp_dist, k=10)
    
    print(f"  Average Distortion (AD):     {hyp_metrics['avg_distortion']:.4f}")
    print(f"  Worst-case Distortion (WD):  {hyp_metrics['worst_distortion']:.4f}")
    print(f"  Std Distortion:              {hyp_metrics['std_distortion']:.4f}")
    print(f"  MAP@10:                      {hyp_map:.4f}")
    
    # 对比结果
    print("\n4. Comparison:")
    print(f"  AD Improvement:  {(1 - hyp_metrics['avg_distortion']/eucl_metrics['avg_distortion'])*100:.2f}%")
    print(f"  WD Improvement:  {(1 - hyp_metrics['worst_distortion']/eucl_metrics['worst_distortion'])*100:.2f}%")
    print(f"  MAP Improvement: {(hyp_map - eucl_map)*100:.2f}%")
    
    return {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'euclidean': {**eucl_metrics, 'map': eucl_map},
        'hyperbolic': {**hyp_metrics, 'map': hyp_map}
    }


def main():
    print("Graph Embedding Distortion Analysis")
    print("Hyperbolic Space Only")
    
    datasets = ['PubMed']
    dim = 128  # 嵌入维度
    c = 0.5    # 双曲空间曲率
    
    results = []
    for dataset_name in datasets:
        result = analyze_dataset(dataset_name, dim=dim, c=c)
        results.append(result)
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("SUMMARY TABLE - Hyperbolic Space")
    print("="*80)
    print(f"{'Dataset':<12} {'Nodes':<8} {'Edges':<8} {'AD':<10} {'WD':<10} {'MAP@10':<10}")
    print("-"*80)
    
    for res in results:
        print(f"{res['dataset']:<12} "
              f"{res['num_nodes']:<8} "
              f"{res['num_edges']:<8} "
              f"{res['hyperbolic']['avg_distortion']:<10.4f} "
              f"{res['hyperbolic']['worst_distortion']:<10.4f} "
              f"{res['hyperbolic']['map']:<10.4f}")
    print("-"*80)


if __name__ == "__main__":
    main()