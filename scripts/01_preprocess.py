# -*- coding: utf-8 -*-
"""
01_preprocess.py
基于你的真实三份文件结构（GSE3431.csv / Node_Labels.csv / Static_PPIN.csv）
完成：ID对齐、表达按Node聚合、z-score标准化、时间切分、dataset子图导出。
运行：python scripts/01_preprocess_dataset.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = ROOT / "outputs"
CONF = ROOT / "conf"
CACHE = DATA / "_cache"

for p in [OUT, CONF, CACHE, OUT/"results"]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- 1) 读取三份文件 ----------
# 表达矩阵：GSE3431.csv，列为 AtID, Node, T1..T36
expr_path = DATA / "GSE3431.csv"
labels_path = DATA / "Node_Labels.csv"
edges_path = DATA / "Static_PPIN.csv"

X_raw = pd.read_csv(expr_path)
labels_raw = pd.read_csv(labels_path)
edges_raw = pd.read_csv(edges_path)

# ---------- 2) 标准化列名（小写）并做字段映射 ----------
def lower_cols(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    return df

X_raw = lower_cols(X_raw)          # 期望有 'node','t1'..'t36'
labels_raw = lower_cols(labels_raw)  # 期望有 'node','label'
edges_raw = lower_cols(edges_raw)    # 期望有 'source','target','weight'(可选)

# 映射/兜底：表达的 Node 列
node_col_expr = "node" if "node" in X_raw.columns else (
    "gene" if "gene" in X_raw.columns else None
)
assert node_col_expr is not None, "GSE3431.csv 中未找到 Node 列（应为 'Node'）"

# 时间列（T1..T36）：按排序自动识别
time_cols = [c for c in X_raw.columns if c.startswith("t")]
time_cols = sorted(time_cols, key=lambda x: int(x[1:]))  # T1..T36 按数字排序
assert len(time_cols) == 36, f"期望 36 个时间列，当前检测到 {len(time_cols)} 个：{time_cols}"

# 标签文件列
node_col_lab = "node" if "node" in labels_raw.columns else None
label_col_lab = "label" if "label" in labels_raw.columns else None
assert node_col_lab and label_col_lab, "Node_Labels.csv 需包含 Node 和 Label 两列"

# 静态边文件列
u_col = "source" if "source" in edges_raw.columns else (
    "src" if "src" in edges_raw.columns else None
)
v_col = "target" if "target" in edges_raw.columns else (
    "dst" if "dst" in edges_raw.columns else None
)
w_col = "weight" if "weight" in edges_raw.columns else None
assert u_col and v_col, "Static_PPIN.csv 需包含 Source 与 Target 列"

# ---------- 3) 表达按 Node 聚合（探针 -> 基因/蛋白） ----------
# 说明：GSE3431 有重复 Node（多探针），这里对每个 Node 的 36 个时间点取均值
X_group = (
    X_raw[[node_col_expr] + time_cols]
    .groupby(node_col_expr, as_index=True)
    .mean()
)
# 去除任何全NaN的节点
X_group = X_group.dropna(how="all")
# 统一为字符串索引
X_group.index = X_group.index.astype(str)

# ---------- 4) dataset 节点集合：来自静态边的端点 ----------
haz_nodes = set(edges_raw[u_col].astype(str)).union(set(edges_raw[v_col].astype(str)))

# 与表达的交集（确保这些节点在表达矩阵中都有时序）
common_nodes = sorted([n for n in haz_nodes if n in X_group.index])

# 过滤表达矩阵到 dataset 子集
X_haz = X_group.loc[common_nodes].copy()

# ---------- 5) z-score 标准化 + 一阶差分 ----------
# 行向 z-score（每个节点在36个时间点上做标准化）
X_z = X_haz.apply(lambda r: (r - r.mean()) / (r.std(ddof=1) + 1e-8), axis=1)
X_delta = X_z.diff(axis=1).fillna(0.0)

# 缓存保存
CACHE.mkdir(parents=True, exist_ok=True)
X_z.to_csv(CACHE / "X_zscore.tsv", sep="\t")
X_delta.to_csv(CACHE / "X_delta.tsv", sep="\t")
np.save(CACHE / "X_zscore.npy", X_z.values)
np.save(CACHE / "X_delta.npy", X_delta.values)

# ---------- 6) 生成 dataset 的静态边子图（只保留两端均在 common_nodes 的边） ----------
E_haz = edges_raw[[u_col, v_col]].astype(str)
E_haz = E_haz[E_haz[u_col].isin(common_nodes) & E_haz[v_col].isin(common_nodes)].drop_duplicates()

# 导出边与节点清单
E_haz.to_csv(OUT / "results/dataset_static_edges.tsv", sep="\t", index=False)
pd.DataFrame({"Node": common_nodes}).to_csv(OUT / "results/dataset_nodes.tsv", sep="\t", index=False)

# ---------- 7) 标签表对齐到 dataset 子集（若有则保存） ----------
labels_sub = labels_raw[[node_col_lab, label_col_lab]].copy()
labels_sub[node_col_lab] = labels_sub[node_col_lab].astype(str)
labels_haz = labels_sub[labels_sub[node_col_lab].isin(common_nodes)].drop_duplicates()
labels_haz.to_csv(OUT / "results/dataset_labels.tsv", sep="\t", index=False)

# ---------- 8) 时间切分配置（T=36） ----------
split = {"train_T": 28, "val_T": 4, "test_T": 4, "total_T": 36}
with open(CONF / "splits_dataset.json", "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2, ensure_ascii=False)

# ---------- 9) 统计信息与一致性报告 ----------
report = {
    "expr_rows_raw": int(len(X_raw)),
    "expr_unique_nodes": int(X_group.shape[0]),
    "dataset_nodes_in_edges": int(len(haz_nodes)),
    "dataset_nodes_with_expr": int(len(common_nodes)),
    "edges_raw": int(len(edges_raw)),
    "edges_dataset_subgraph": int(len(E_haz)),
    "time_points": 36,
    "labels_total": int(len(labels_raw)),
    "labels_on_dataset": int(len(labels_haz)),
    "columns_GSE3431": [c for c in X_raw.columns],
    "columns_Node_Labels": [c for c in labels_raw.columns],
    "columns_Static_PPIN": [c for c in edges_raw.columns]
}
with open(OUT / "results/dataset_preprocess_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\n已完成：dataset 节点 {len(common_nodes)} ｜ 边 {len(E_haz)} ｜ T=36")
print(f"缓存：{CACHE/'X_zscore.tsv'} / {CACHE/'X_delta.tsv'}")
print(f"子图：{OUT/'results/dataset_static_edges.tsv'}，标签：{OUT/'results/dataset_labels.tsv'}")
