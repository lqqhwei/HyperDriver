# src/data_utils.py

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================
# Configuration & Path Management
# ============================

@dataclass
class DatasetPaths:
    """Description of the original file paths for a single dataset"""
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
    Read the list of enabled datasets from conf/datasets.json.
    Only return the name where enabled == true. :contentReference[oaicite:8]{index=8}
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
    Construct the path for a dataset. The root_dir is typically D:/HYPERDRIVER.
    """
    data_dir = os.path.join(root_dir, "data", dataset_name)
    return DatasetPaths(name=dataset_name, root=root_dir, data_dir=data_dir)


# ============================
# Label & Node feature deal with
# ============================

def load_global_labels(root_dir: str) -> pd.DataFrame:
    """
    Read the global Node_Labels_with_essential.csv file.
    Your actual file columns will likely be:
    ['Node', 'Label', 'SGD', 'OGEE', 'DEG', 'SOD', 'essential']
    We treat 'Node' as the protein/ORF ID and 'essential' as the required label. :contentReference[oaicite:9]{index=9}
    """
    label_path = os.path.join(root_dir, "data", "Node_Labels_with_essential.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Global label file not found: {label_path}")

    df = pd.read_csv(label_path)
    col_map = {c.lower(): c for c in df.columns}

    # protein / gene id Column: Compatibility 'Node', 'protein', 'orf', 'gene'
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

    # Conservative approach: Non-zero values ​​are considered necessary.
    df["essential"] = (df["essential"] != 0).astype(int)
    return df


def load_node_features(path: str) -> pd.DataFrame:
    """
    Read Node_Features.txt。:contentReference[oaicite:10]{index=10}

    Based on the Hazbun snippet you provided, the format is comma-separated:
        idx, ORF, t1, t2, ..., t36

    Therefore, it is read as a CSV file (header=None) and automatically named:
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
    Build an index <-> protein mapping from Node_Features.
    return:
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
    Merge Node_Features with the global Essential tag by protein.
    Proteins that do not appear in the label table are assigned an essential value of 0 by default.
    """
    nodes_df = node_feat_df.copy()
    labels_df = global_labels_df.copy()

    merged = nodes_df.merge(labels_df, on="protein", how="left")
    merged["essential"] = merged["essential"].fillna(0).astype(int)
    return merged


# ============================
# Edge list reading
# ============================

def load_static_edges(path: str) -> pd.DataFrame:
    """
    Read Static_PPIN.txt. :contentReference[oaicite:11]{index=11}

    Judging from the Hazbun clip, the format is:
        source_protein   target_protein   weight
    Use any whitespace as a separator.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"Unexpected Static_PPIN format, cols={df.shape[1]}, path={path}")
    df = df.iloc[:, :3]
    df.columns = ["src", "dst", "weight"]
    return df


def load_dynamic_edges(path: str) -> pd.DataFrame:
    """
    Read Dynamic_PPIN.txt。:contentReference[oaicite:12]{index=12}

    Judging from the fragment, the format is comma-separated:
        src_idx, dst_idx, t, weight
    For added security, use regular expressions that are compatible with commas or whitespace.
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
    Map the protein IDs in Static_PPIN to indices (consistent with Node_Features).
    """
    df = static_df.copy()
    df["src_idx"] = df["src"].map(protein_to_index)
    df["dst_idx"] = df["dst"].map(protein_to_index)

    # Filter out edges that cannot be mapped
    df = df.dropna(subset=["src_idx", "dst_idx"])
    df["src_idx"] = df["src_idx"].astype(int)
    df["dst_idx"] = df["dst_idx"].astype(int)
    return df[["src_idx", "dst_idx", "weight"]]


# ============================
# Unified preprocessing entry point
# ============================

def preprocess_single_dataset(root_dir: str, dataset_name: str) -> None:
    """
    Preprocessing a single dataset:
    - Read Node_Features, Static_PPIN, Dynamic_PPIN
    - Align global essential labels
    - Generate processed/<dataset>/nodes.csv, static_edges.csv, dynamic_edges.csv
    """
    paths = get_dataset_paths(root_dir, dataset_name)

    # 1) Load node features & mapping
    node_feat_df = load_node_features(paths.node_features)
    index_to_protein, protein_to_index = build_index_mapping(node_feat_df)

    # 2) Load the global Essential tags and merge them into the node table.
    global_labels_df = load_global_labels(root_dir)
    nodes_merged = merge_features_and_labels(node_feat_df, global_labels_df)

    # 3) Load static edges and map them to index.
    static_df = load_static_edges(paths.static_ppin)
    static_idx_df = map_static_edges_to_indices(static_df, protein_to_index)

    # 4) Load dynamic edges (already in index form)
    dyn_df = load_dynamic_edges(paths.dynamic_ppin)

    # 5) Save to the processed/<dataset> directory
    os.makedirs(paths.processed_dir, exist_ok=True)
    nodes_out = os.path.join(paths.processed_dir, "nodes.csv")
    static_out = os.path.join(paths.processed_dir, "static_edges.csv")
    dynamic_out = os.path.join(paths.processed_dir, "dynamic_edges.csv")

    nodes_merged.to_csv(nodes_out, index=False)
    static_idx_df.to_csv(static_out, index=False)
    dyn_df.to_csv(dynamic_out, index=False)

    print(f"[OK] Preprocessed dataset {dataset_name} -> {paths.processed_dir}")
