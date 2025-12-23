# train_hyperdriver.py
"""
Step 3: 训练 HyperDriver 模型 (Clean & Fixed Version)
功能：
1. 只训练 "full" 变体 (配合 V16+ 推理消融策略)。
2. 包含完整的鲁棒数据加载逻辑 (索引越界检查)。
3. 修复了之前的语法错误。
"""

import argparse
import os
import random
import traceback
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_utils import load_datasets_config
from src.hyper_driver import HyperDriver, HyperDriverConfig

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 数据构建函数
# ============================

def build_inputs_with_alignment(
    root_dir: str,
    dataset_name: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], torch.Tensor, pd.DataFrame]:
    """
    构造模型输入，并将动态图(Teacher)的权重对齐到静态骨架上。
    """
    proc_dir = os.path.join(root_dir, "processed", dataset_name)
    nodes_path = os.path.join(proc_dir, "nodes.csv")
    static_edges_path = os.path.join(proc_dir, "static_edges.csv")
    dynamic_edges_path = os.path.join(proc_dir, "dynamic_edges.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(static_edges_path):
        raise FileNotFoundError(f"Missing processed files for {dataset_name}")

    # 1. Nodes & Labels
    nodes_df = pd.read_csv(nodes_path)
    N = len(nodes_df)
    labels = torch.tensor(nodes_df["essential"].values, dtype=torch.float, device=device)

    # 2. Features
    time_cols = sorted([c for c in nodes_df.columns if c.startswith("t") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    if not time_cols:
        # Fallback if no time features
        x_seq_all = torch.zeros(N, 1, 1, device=device)
        T_feat = 1
    else:
        x_list = [torch.tensor(nodes_df[c].values.astype(np.float32), device=device).unsqueeze(-1) for c in time_cols]
        x_seq_all = torch.stack(x_list, dim=0) # [T, N, 1]
        T_feat = len(time_cols)

    # 3. Static Edges
    static_df = pd.read_csv(static_edges_path)
    src_s = static_df["src_idx"].values.astype(np.int64)
    dst_s = static_df["dst_idx"].values.astype(np.int64)
    
    # Filter invalid edges
    valid_mask = (src_s >= 0) & (src_s < N) & (dst_s >= 0) & (dst_s < N)
    if not valid_mask.all():
        print(f"[WARN] Filtering {(~valid_mask).sum()} invalid edges in {dataset_name}")
        src_s = src_s[valid_mask]
        dst_s = dst_s[valid_mask]
    
    static_edge_index = torch.tensor(np.stack([src_s, dst_s]), dtype=torch.long, device=device)
    E_static = static_edge_index.size(1)
    
    # Build edge map for teacher alignment
    static_edge_map = {(int(src_s[i]), int(dst_s[i])): i for i in range(E_static)}
    teacher_weights_list = []

    # 4. Dynamic Edges (Teacher)
    if os.path.exists(dynamic_edges_path):
        dyn_df = pd.read_csv(dynamic_edges_path)
        if not dyn_df.empty:
            t_vals = sorted(dyn_df["t"].unique())
            # T is min of features and dynamic graphs
            T = min(T_feat, len(t_vals))
            x_seq = x_seq_all[:T]
            
            for t_val in t_vals[:T]:
                w_target = torch.zeros(E_static, dtype=torch.float, device=device)
                sub = dyn_df[dyn_df["t"] == t_val]
                d_src = sub["src_idx"].values
                d_dst = sub["dst_idx"].values
                d_w = sub["weight"].values
                
                for i in range(len(d_src)):
                    u, v, w = int(d_src[i]), int(d_dst[i]), float(d_w[i])
                    if u >= N or v >= N or u < 0 or v < 0: continue
                    
                    if (u, v) in static_edge_map:
                        w_target[static_edge_map[(u, v)]] = w
                    elif (v, u) in static_edge_map: # Undirected fallback
                        w_target[static_edge_map[(v, u)]] = w
                        
                teacher_weights_list.append(w_target)
        else:
            T = T_feat
            x_seq = x_seq_all
            teacher_weights_list = [torch.zeros(E_static, device=device) for _ in range(T_feat)]
    else:
        T = T_feat
        x_seq = x_seq_all
        teacher_weights_list = [torch.zeros(E_static, device=device) for _ in range(T_feat)]

    return x_seq, static_edge_index, teacher_weights_list, labels, nodes_df

# ============================
# 训练主逻辑 (Full Model Only)
# ============================

def train_full_model(
    root_dir: str,
    dataset_name: str,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    seed: int = 42
):
    """只训练 Full Model"""
    device = get_device()
    set_seed(seed)
    
    print(f"\n========== Training {dataset_name} (FULL MODEL) ==========")
    
    # 1. Load Data
    try:
        x_seq, static_edge_index, teacher_weights_list, labels, nodes_df = \
            build_inputs_with_alignment(root_dir, dataset_name, device)
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return
    
    N = x_seq.shape[1]
    
    # 2. Split
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)
    split = int(0.8 * N)
    train_idx = torch.tensor(all_idx[:split], dtype=torch.long, device=device)
    val_idx = torch.tensor(all_idx[split:], dtype=torch.long, device=device)

    # 3. Model
    config = HyperDriverConfig(
        in_feats=x_seq.shape[2],
        hidden_dim=hidden_dim,
        num_graph_layers=2,
        num_hyper_layers=1,
        num_time_layers=1,
        num_classes=1,
        num_scales=[5, 10, 15], # Full model uses multi-scale
        dropout=0.1
    )
    model = HyperDriver(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    crit_task = nn.BCEWithLogitsLoss()
    crit_distill = nn.MSELoss()

    # 4. Loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(x_seq, static_edge_index)
        logits = out["logits"].squeeze(-1)
        z_mix = out["z_mix"]
        
        # Loss 1: Task
        loss_task = crit_task(logits[train_idx], labels[train_idx])
        
        # Loss 2: Distill
        loss_distill = 0.0
        if len(out["predicted_weights_list"]) > 0:
            loss_distill = crit_distill(torch.stack(out["predicted_weights_list"]), torch.stack(teacher_weights_list))
            
        # Loss 3: Stability
        loss_stab = torch.norm(z_mix, p='fro') / N
        
        # Total Loss
        loss = loss_task + 1.0 * loss_distill + 0.01 * loss_stab
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_acc = ((torch.sigmoid(logits[val_idx]) > 0.5) == labels[val_idx]).float().mean()
            print(f"[Ep {epoch}] Loss={loss.item():.4f} | Val Acc={val_acc:.4f}")

    # 5. Save Checkpoint (Only Full)
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{dataset_name}_full.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "variant": "full"
    }, path)
    print(f"[INFO] Model saved: {path}")
    
    # 6. Save Scores
    res_dir = os.path.join(root_dir, "results", dataset_name, "full")
    os.makedirs(res_dir, exist_ok=True)
    nodes_df["prob"] = torch.sigmoid(logits).detach().cpu().numpy()
    nodes_df.to_csv(os.path.join(res_dir, "node_scores.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    # Load dataset list
    conf_path = os.path.join(root, "conf", "datasets.json")
    if args.dataset == "all":
        datasets = load_datasets_config(conf_path)
    else:
        datasets = [args.dataset]

    print(f"========== HyperDriver Training (Clean) ==========")
    for ds in datasets:
        try:
            train_full_model(root, ds, epochs=args.epochs)
        except Exception as e:
            print(f"[ERR] {ds}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()