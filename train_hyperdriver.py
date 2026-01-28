import argparse
import os
import random
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

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

def build_inputs_structure_only(root_dir: str, dataset_name: str, device: torch.device):
    """
    Construct model input without any label dependency.
    """
    proc_dir = os.path.join(root_dir, "processed", dataset_name)
    nodes_df = pd.read_csv(os.path.join(proc_dir, "nodes.csv"))
    static_df = pd.read_csv(os.path.join(proc_dir, "static_edges.csv"))
    
    N = len(nodes_df)
    
    # Features
    time_cols = sorted([c for c in nodes_df.columns if c.startswith("t") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    if not time_cols:
        x_seq_all = torch.zeros(N, 1, 1, device=device)
        T_feat = 1
    else:
        x_list = [torch.tensor(nodes_df[c].values.astype(np.float32), device=device).unsqueeze(-1) for c in time_cols]
        x_seq_all = torch.stack(x_list, dim=0)
        T_feat = len(time_cols)

    # Static Graph
    src_s = static_df["src_idx"].values.astype(np.int64)
    dst_s = static_df["dst_idx"].values.astype(np.int64)
    static_edge_index = torch.tensor(np.stack([src_s, dst_s]), dtype=torch.long, device=device)
    
    # Dynamic Graph (Teacher) construction logic...
    # (Simplified for brevity: Assume standard teacher weights loading or construction here)
    # Re-using the logic from your original file for Teacher Weights:
    E_static = static_edge_index.size(1)
    static_edge_map = {(int(src_s[i]), int(dst_s[i])): i for i in range(E_static)}
    teacher_weights_list = []
    
    dyn_path = os.path.join(proc_dir, "dynamic_edges.csv")
    if os.path.exists(dyn_path):
        dyn_df = pd.read_csv(dyn_path)
        t_vals = sorted(dyn_df["t"].unique())
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
                if (u,v) in static_edge_map: w_target[static_edge_map[(u,v)]] = w
                elif (v,u) in static_edge_map: w_target[static_edge_map[(v,u)]] = w
            teacher_weights_list.append(w_target)
    else:
        # Fallback if no dynamic edges
        x_seq = x_seq_all
        teacher_weights_list = [torch.zeros(E_static, device=device) for _ in range(T_feat)]

    return x_seq, static_edge_index, teacher_weights_list, nodes_df

def train_structure_model(root_dir: str, dataset_name: str, epochs: int = 100):
    device = get_device()
    set_seed(42)
    print(f"\n========== Training {dataset_name} (Structure Only) ==========")

    try:
        x_seq, static_edge_index, teacher_weights_list, nodes_df = \
            build_inputs_structure_only(root_dir, dataset_name, device)
    except Exception as e:
        print(f"[ERROR] {e}"); return

    # Model Configuration (Num Classes is irrelevant but kept for arch compatibility)
    config = HyperDriverConfig(
        in_feats=x_seq.shape[2], hidden_dim=64, num_graph_layers=2, 
        num_hyper_layers=1, num_time_layers=1, num_classes=1, num_scales=[5, 10, 15], dropout=0.1
    )
    model = HyperDriver(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit_distill = nn.MSELoss()

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_seq, static_edge_index)
        
        # Loss: Distillation + Stability Only
        loss_distill = 0.0
        if len(out["predicted_weights_list"]) > 0:
            loss_distill = crit_distill(torch.stack(out["predicted_weights_list"]), torch.stack(teacher_weights_list))
        loss = 1.0 * loss_distill + 0.01 * (torch.norm(out["z_mix"], p='fro') / x_seq.shape[1])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"[Ep {epoch}] Loss={loss.item():.4f}")

    # Save Checkpoint & Results
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, 
               os.path.join(ckpt_dir, f"{dataset_name}_full.pt"))
    
    # Save base node list for eval_driver
    res_dir = os.path.join(root_dir, "results", dataset_name, "full")
    os.makedirs(res_dir, exist_ok=True)
    nodes_df.to_csv(os.path.join(res_dir, "node_scores.csv"), index=False)
    print(f"[INFO] Finished {dataset_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    root = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(root, "conf", "datasets.json")
    datasets = load_datasets_config(conf_path) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        train_structure_model(root, ds, epochs=args.epochs)

if __name__ == "__main__":
    main()