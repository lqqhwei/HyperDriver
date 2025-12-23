# src/data_utils.py

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================
# 配置 & 路径管理
# ============================

@dataclass
class DatasetPaths:
    """单个数据集的原始文件路径描述"""
    name: str
    root: str  # e.g. D:/HYPERDRIVER
    data_dir: str  # e.g. D:/HYPERDRIVER/data/Hazbun

    @property
    def static_ppin(self) -> str:
        return os.path.join(self.data_dir, "Static_PPIN.txt")

    @property
    def dynamic_ppin(self) -> str:
        return os.path.join(self.data_dir, "Dynamic_PPIN.txt")

    @property
    def node_features(self) -> str:
        return os.path.join(self.data_dir, "Node_Features.txt")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed", self.name)


def load_datasets_config(conf_path: str) -> List[str]:
    """
    从 conf/datasets.json 读取启用的数据集列表。
    只返回 enabled == true 的 name。:contentReference[oaicite:8]{index=8}
    """
    with open(conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    enabled = []
    for item in cfg.get("datasets", []):
        if item.get("enabled", False):
            enabled.append(item["name"])
    return enabled


def get_dataset_paths(root_dir: str, dataset_name: str) -> DatasetPaths:
    """
    构造某个数据集的路径。root_dir 通常为 D:/HYPERDRIVER
    """
    data_dir = os.path.join(root_dir, "data", dataset_name)
    return DatasetPaths(name=dataset_name, root=root_dir, data_dir=data_dir)


# ============================
# Label & Node feature 处理
# ============================

def load_global_labels(root_dir: str) -> pd.DataFrame:
    """
    读取全局的 Node_Labels_with_essential.csv。

    你的实际文件列大概是：
    ['Node', 'Label', 'SGD', 'OGEE', 'DEG', 'SOD', 'essential']
    我们把 'Node' 当成 protein/ORF ID，'essential' 当成必需标记。:contentReference[oaicite:9]{index=9}
    """
    label_path = os.path.join(root_dir, "data", "Node_Labels_with_essential.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Global label file not found: {label_path}")

    df = pd.read_csv(label_path)
    col_map = {c.lower(): c for c in df.columns}

    # protein / gene id 列：兼容 'Node', 'protein', 'orf', 'gene'
    protein_col = None
    for key in ["node", "protein", "orf", "gene", "label"]:
        if key in col_map:
            protein_col = col_map[key]
            break
    if protein_col is None:
        raise ValueError(
            f"Cannot find protein/gene column in {label_path}, "
            f"found columns = {list(df.columns)}"
        )

    # essential 列
    ess_col = None
    for key in ["essential", "is_essential", "essential_flag"]:
        if key in col_map:
            ess_col = col_map[key]
            break
    if ess_col is None:
        raise ValueError(
            f"Cannot find essential label column in {label_path}, got {list(df.columns)}"
        )

    df = df[[protein_col, ess_col]].copy()
    df.columns = ["protein", "essential"]

    # 保守处理：非 0 视为必需
    df["essential"] = (df["essential"] != 0).astype(int)
    return df


def load_node_features(path: str) -> pd.DataFrame:
    """
    读取 Node_Features.txt。:contentReference[oaicite:10]{index=10}

    从你提供的 Hazbun 片段看，格式是逗号分隔：
        idx, ORF, t1, t2, ..., t36

    所以这里按 CSV（header=None）读取，并自动命名：
        index, protein, t1..tN
    """
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    if n_cols < 3:
        raise ValueError(f"Unexpected Node_Features format, columns = {n_cols}, path={path}")

    col_names = ["index", "protein"] + [f"t{i+1}" for i in range(n_cols - 2)]
    df.columns = col_names
    return df


def build_index_mapping(node_feat_df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    从 Node_Features 里构建 index <-> protein 映射。
    返回：
        index_to_protein: {0: 'YER127W', ...}
        protein_to_index: {'YER127W': 0, ...}
    """
    if "index" not in node_feat_df.columns or "protein" not in node_feat_df.columns:
        raise ValueError("node_feat_df must contain 'index' and 'protein' columns.")

    index_to_protein = dict(zip(node_feat_df["index"].astype(int), node_feat_df["protein"].astype(str)))
    protein_to_index = {p: idx for idx, p in index_to_protein.items()}
    return index_to_protein, protein_to_index


def merge_features_and_labels(
    node_feat_df: pd.DataFrame,
    global_labels_df: pd.DataFrame
) -> pd.DataFrame:
    """
    把 Node_Features 与全局 Essential 标签按 protein 合并。
    未出现在 label 表中的蛋白，默认 essential=0。
    """
    nodes_df = node_feat_df.copy()
    labels_df = global_labels_df.copy()

    merged = nodes_df.merge(labels_df, on="protein", how="left")
    merged["essential"] = merged["essential"].fillna(0).astype(int)
    return merged


# ============================
# 边列表读取
# ============================

def load_static_edges(path: str) -> pd.DataFrame:
    """
    读取 Static_PPIN.txt。:contentReference[oaicite:11]{index=11}

    从 Hazbun 片段看，格式是：
        source_protein   target_protein   weight
    使用任意空白分隔。
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"Unexpected Static_PPIN format, cols={df.shape[1]}, path={path}")
    df = df.iloc[:, :3]
    df.columns = ["src", "dst", "weight"]
    return df


def load_dynamic_edges(path: str) -> pd.DataFrame:
    """
    读取 Dynamic_PPIN.txt。:contentReference[oaicite:12]{index=12}

    从片段看，格式为逗号分隔：
        src_idx, dst_idx, t, weight
    为了更稳妥，使用正则兼容逗号或空白。
    """
    df = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None)
    if df.shape[1] < 4:
        raise ValueError(f"Unexpected Dynamic_PPIN format, cols={df.shape[1]}, path={path}")
    df = df.iloc[:, :4]
    df.columns = ["src_idx", "dst_idx", "t", "weight"]
    return df


def map_static_edges_to_indices(
    static_df: pd.DataFrame,
    protein_to_index: Dict[str, int]
) -> pd.DataFrame:
    """
    把 Static_PPIN 中的 protein ID 映射成 index（与 Node_Features 保持一致）。
    """
    df = static_df.copy()
    df["src_idx"] = df["src"].map(protein_to_index)
    df["dst_idx"] = df["dst"].map(protein_to_index)

    # 过滤掉无法映射的边
    df = df.dropna(subset=["src_idx", "dst_idx"])
    df["src_idx"] = df["src_idx"].astype(int)
    df["dst_idx"] = df["dst_idx"].astype(int)
    return df[["src_idx", "dst_idx", "weight"]]


# ============================
# 统一预处理入口
# ============================

def preprocess_single_dataset(root_dir: str, dataset_name: str) -> None:
    """
    预处理单个数据集：
    - 读取 Node_Features, Static_PPIN, Dynamic_PPIN
    - 对齐全局 essential 标签
    - 生成 processed/<dataset>/nodes.csv, static_edges.csv, dynamic_edges.csv
    """
    paths = get_dataset_paths(root_dir, dataset_name)

    # 1) 加载节点特征 & 映射
    node_feat_df = load_node_features(paths.node_features)
    index_to_protein, protein_to_index = build_index_mapping(node_feat_df)

    # 2) 加载全局 Essential 标签，并合并到节点表
    global_labels_df = load_global_labels(root_dir)
    nodes_merged = merge_features_and_labels(node_feat_df, global_labels_df)

    # 3) 加载静态边，并映射到 index
    static_df = load_static_edges(paths.static_ppin)
    static_idx_df = map_static_edges_to_indices(static_df, protein_to_index)

    # 4) 加载动态边（已经是 index 形式）
    dyn_df = load_dynamic_edges(paths.dynamic_ppin)

    # 5) 保存到 processed/<dataset> 目录
    os.makedirs(paths.processed_dir, exist_ok=True)
    nodes_out = os.path.join(paths.processed_dir, "nodes.csv")
    static_out = os.path.join(paths.processed_dir, "static_edges.csv")
    dynamic_out = os.path.join(paths.processed_dir, "dynamic_edges.csv")

    nodes_merged.to_csv(nodes_out, index=False)
    static_idx_df.to_csv(static_out, index=False)
    dyn_df.to_csv(dynamic_out, index=False)

    print(f"[OK] Preprocessed dataset {dataset_name} -> {paths.processed_dir}")
