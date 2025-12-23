# baselines_centrality.py
"""
Step 5: 计算基准中心性 (DC, BC, EC)

[Bold Upgrade]: 
仅负责计算并保存基准分数，移除所有旧的"覆盖率对比"逻辑。
真正的"能量大乱斗"评测将统一在 eval_driver.py 中进行。
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from src.data_utils import load_datasets_config


# ============================
# 中心性算法 (Standard)
# ============================

def compute_degree_centrality(N: int, edges_u: np.ndarray, edges_v: np.ndarray,
                              weights: np.ndarray) -> np.ndarray:
    deg = np.zeros(N, dtype=np.float64)
    for u, v, w in zip(edges_u, edges_v, weights):
        deg[u] += w
        deg[v] += w
    return deg

def compute_eigenvector_centrality(
    N: int,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    neighbors: List[List[int]] = [[] for _ in range(N)]
    w_list: List[List[float]] = [[] for _ in range(N)]
    for u, v, w in zip(edges_u, edges_v, weights):
        neighbors[u].append(int(v))
        w_list[u].append(float(w))
        neighbors[v].append(int(u))
        w_list[v].append(float(w))

    rng = np.random.default_rng(2025)
    x = rng.normal(size=N)
    x = np.abs(x)
    x /= (np.linalg.norm(x) + 1e-8)

    for _ in range(max_iter):
        y = np.zeros_like(x)
        for i in range(N):
            s = 0.0
            for j, w_ij in zip(neighbors[i], w_list[i]):
                s += w_ij * x[j]
            y[i] = s
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-12: break
        y /= norm_y
        if np.linalg.norm(y - x) < tol: break
        x = y
    return x

def compute_betweenness_centrality_unweighted(
    N: int,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
) -> np.ndarray:
    neighbors: List[List[int]] = [[] for _ in range(N)]
    for u, v in zip(edges_u, edges_v):
        neighbors[u].append(int(v))
        neighbors[v].append(int(u))

    BC = np.zeros(N, dtype=np.float64)

    for s in range(N):
        stack = []
        pred: List[List[int]] = [[] for _ in range(N)]
        sigma = np.zeros(N, dtype=np.float64)
        sigma[s] = 1.0
        dist = -np.ones(N, dtype=np.int64)
        dist[s] = 0
        queue = [s]
        
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in neighbors[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = np.zeros(N, dtype=np.float64)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    delta[v] += delta_v
            if w != s:
                BC[w] += delta[w]

    BC /= 2.0
    return BC

# ============================
# 处理流程
# ============================

def process_one_dataset(root_dir: str, dataset_name: str):
    print(f"\n========== Computing Baselines for {dataset_name} ==========")

    proc_dir = os.path.join(root_dir, "processed", dataset_name)
    nodes_path = os.path.join(proc_dir, "nodes.csv")
    static_edges_path = os.path.join(proc_dir, "static_edges.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(static_edges_path):
        print(f"[WARN] Files missing for {dataset_name}")
        return

    nodes_df = pd.read_csv(nodes_path)
    static_df = pd.read_csv(static_edges_path)
    N = len(nodes_df)
    
    # 构造图
    u = static_df["src_idx"].to_numpy(dtype=np.int64)
    v = static_df["dst_idx"].to_numpy(dtype=np.int64)
    w = static_df["weight"].to_numpy(dtype=np.float64)

    u2 = np.minimum(u, v)
    v2 = np.maximum(u, v)
    edges_df = pd.DataFrame({"u": u2, "v": v2, "weight": w})
    edges_df = edges_df.groupby(["u", "v"], as_index=False)["weight"].sum()

    u_final = edges_df["u"].to_numpy(dtype=np.int64)
    v_final = edges_df["v"].to_numpy(dtype=np.int64)
    w_final = edges_df["weight"].to_numpy(dtype=np.float64)

    # 计算分数
    print(f"[INFO] Computing Degree, Eigenvector, Betweenness for N={N}...")
    dc = compute_degree_centrality(N, u_final, v_final, w_final)
    ec = compute_eigenvector_centrality(N, u_final, v_final, w_final)
    bc = compute_betweenness_centrality_unweighted(N, u_final, v_final)

    # 保存
    base_dir = os.path.join(root_dir, "results", dataset_name, "baselines")
    os.makedirs(base_dir, exist_ok=True)

    # 保存时带上 protein ID 方便后续合并
    base_df = nodes_df[["index", "protein"]].copy()
    
    # 分别保存，互不干扰
    pd.concat([base_df, pd.DataFrame({"dc": dc})], axis=1).to_csv(os.path.join(base_dir, "dc_scores.csv"), index=False)
    pd.concat([base_df, pd.DataFrame({"ec": ec})], axis=1).to_csv(os.path.join(base_dir, "ec_scores.csv"), index=False)
    pd.concat([base_df, pd.DataFrame({"bc": bc})], axis=1).to_csv(os.path.join(base_dir, "bc_scores.csv"), index=False)
    
    print(f"[INFO] Baseline scores saved to {base_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == "all":
        dataset_list = load_datasets_config(os.path.join(root_dir, "conf", "datasets.json"))
    else:
        dataset_list = [args.dataset]

    for ds in dataset_list:
        process_one_dataset(root_dir, ds)

if __name__ == "__main__":
    main()