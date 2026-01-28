import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

# ============================
# Configuration & Path Management
# ============================

@dataclass
class DatasetPaths:
    name: str
    root: str
    data_dir: str

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
    with open(conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [item["name"] for item in cfg.get("datasets", []) if item.get("enabled", False)]


def get_dataset_paths(root_dir: str, dataset_name: str) -> DatasetPaths:
    data_dir = os.path.join(root_dir, "data", dataset_name)
    return DatasetPaths(name=dataset_name, root=root_dir, data_dir=data_dir)


# ============================
# Node Feature Loading (Master Source)
# ============================

def load_node_features(path: str) -> pd.DataFrame:
    """Reads Node_Features.txt. This file is now the ONLY source of node info."""
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    # Standard format: index, protein, t1, t2...
    col_names = ["index", "protein"] + [f"t{i+1}" for i in range(n_cols - 2)]
    df.columns = col_names
    return df


def build_index_mapping(node_feat_df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int]]:
    index_to_protein = dict(zip(node_feat_df["index"].astype(int), node_feat_df["protein"].astype(str)))
    protein_to_index = {p: idx for idx, p in index_to_protein.items()}
    return index_to_protein, protein_to_index


# ============================
# Edge Loading
# ============================

def load_static_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :3]
    df.columns = ["src", "dst", "weight"]
    return df


def load_dynamic_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None)
    df = df.iloc[:, :4]
    df.columns = ["src_idx", "dst_idx", "t", "weight"]
    return df


def map_static_edges_to_indices(static_df: pd.DataFrame, protein_to_index: Dict[str, int]) -> pd.DataFrame:
    df = static_df.copy()
    df["src_idx"] = df["src"].map(protein_to_index)
    df["dst_idx"] = df["dst"].map(protein_to_index)
    df = df.dropna(subset=["src_idx", "dst_idx"])
    df["src_idx"] = df["src_idx"].astype(int)
    df["dst_idx"] = df["dst_idx"].astype(int)
    return df[["src_idx", "dst_idx", "weight"]]


# ============================
# Preprocessing Entry Point
# ============================

def preprocess_single_dataset(root_dir: str, dataset_name: str) -> None:
    paths = get_dataset_paths(root_dir, dataset_name)

    # 1. Nodes (From Features Only) - No Labels merging anymore
    node_feat_df = load_node_features(paths.node_features)
    _, protein_to_index = build_index_mapping(node_feat_df)
    
    # 2. Edges
    static_df = load_static_edges(paths.static_ppin)
    static_idx_df = map_static_edges_to_indices(static_df, protein_to_index)
    dyn_df = load_dynamic_edges(paths.dynamic_ppin)

    # 3. Save
    os.makedirs(paths.processed_dir, exist_ok=True)
    node_feat_df.to_csv(os.path.join(paths.processed_dir, "nodes.csv"), index=False)
    static_idx_df.to_csv(os.path.join(paths.processed_dir, "static_edges.csv"), index=False)
    dyn_df.to_csv(os.path.join(paths.processed_dir, "dynamic_edges.csv"), index=False)

    print(f"[OK] Preprocessed {dataset_name} (Labels Removed)")