#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 SGD 的 phenotype_data.tab 生成必需基因列表，
并给 Node_Labels.csv 增加以下 5 列：

1) SGD  : 该 ORF 是否在 SGD 的 inviabl* 表型列表中 (0/1)
2) OGEE : 是否在 OGEE 必需基因列表中 (0/1)
3) DEG  : 是否在 DEG 必需基因列表中 (0/1)
4) SOD  : SGD + OGEE + DEG 的和
5) essential : 若 SOD >= 1 则为 1，否则为 0

使用前准备：
- 当前目录（或你设置的 BASE_DIR）里有：
    phenotype_data.tab
    Node_Labels.csv
    OGEE_essential_orfs.txt
    DEG_essential_orfs.txt

说明：
- OGEE_essential_orfs.txt / DEG_essential_orfs.txt 格式要求：
    每行一个 ORF（如 YGR129W），无表头。
"""

from pathlib import Path
import csv
import pandas as pd
import shutil

# ---------------- 路径配置（按需修改） ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

PHENO_PATH        = DATA_DIR / "phenotype_data.tab"          # SGD 下载的原始文件
NODE_LABELS_PATH  = DATA_DIR / "Node_Labels.csv"             # 你最开始那张表
OGEE_LIST_PATH    = OUTPUT_DIR / "OGEE_essential_orfs.txt"     # 你已经准备好
DEG_LIST_PATH     = OUTPUT_DIR / "DEG_essential_orfs.txt"      # 你已经准备好

SGD_LIST_PATH     = OUTPUT_DIR / "SGD_essential_orfs.txt"      # 本脚本会自动生成
OUTPUT_NODE_LABELS = OUTPUT_DIR / "Node_Labels_with_essential.csv"   # 本脚本会自动生成

ROOT_DIR = Path(__file__).resolve().parents[3]
ROOT_DATA = ROOT_DIR / "data"

# ---------------- 工具函数 ----------------
def load_orf_set(txt_path: Path) -> set:
    """从 txt 文件载入 ORF 集合。
    要求：每行一个 ORF 名（如 YGR129W），可以有空行，会自动跳过。
    如果一行有多个字段，只取第一个字段。
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到文件: {txt_path}")

    orfs = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            orf = line.split()[0]
            orfs.add(orf)
    return orfs

# ---------------- 第一步：从 phenotype_data.tab 生成 SGD_essential_orfs.txt ----------------
def build_sgd_essential_list(pheno_path: Path, out_path: Path) -> None:
    """
    从 SGD 的 phenotype_data.tab 中筛选出所有 inviabl* 表型的 ORFs，
    生成一列 ORF 的文本文件 out_path。

    说明：
    - 不再使用 pandas.read_csv，而是用 csv.reader 手动解析，
      因为文件中有些行是 14 列，有些是 15 列或 13 列。
    - 这里只依赖三列：
        0: ORF
        1: feature_type
        9: phenotype observable（含 'inviab' 字样）
    """
    if not pheno_path.exists():
        raise FileNotFoundError(f"找不到 phenotype_data.tab: {pheno_path}")

    print(f"[INFO] 读取 phenotype 数据: {pheno_path}")

    essential_orfs = set()
    total_rows = 0
    used_rows = 0

    with pheno_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            total_rows += 1

            # 跳过空行或列数太少的奇怪行
            if not row or len(row) < 10:
                continue

            orf = row[0].strip()
            feature_type = row[1].strip().upper()
            phenotype = row[9].strip().lower()  # 第 9 列是 phenotype 描述

            # 只要 ORF 类型
            if feature_type != "ORF":
                continue

            # 含 inviabl* 的表型视为必需
            if "inviab" in phenotype:
                essential_orfs.add(orf)
                used_rows += 1

    essential_orfs = sorted(essential_orfs)
    print(f"[INFO] 总行数: {total_rows}")
    print(f"[INFO] 符合 inviabl* 记录行数: {used_rows}")
    print(f"[INFO] 不重复 ORF 数量: {len(essential_orfs)}")

    # 保存为一列文本，每行一个 ORF
    with out_path.open("w", encoding="utf-8") as f:
        for orf in essential_orfs:
            f.write(f"{orf}\n")

    print(f"[DONE] 已生成 SGD 必需基因列表: {out_path}")

# ---------------- 第二步：给 Node_Labels.csv 增加 SGD/OGEE/DEG/SOD/essential ----------------
def annotate_node_labels(node_labels_path: Path,
                         sgd_list_path: Path,
                         ogee_list_path: Path,
                         deg_list_path: Path,
                         output_path: Path) -> None:
    """
    读入 Node_Labels.csv，并基于 3 个列表文件增加 5 列：
    SGD, OGEE, DEG, SOD, essential
    """
    if not node_labels_path.exists():
        raise FileNotFoundError(f"找不到 Node_Labels.csv: {node_labels_path}")

    print(f"[INFO] 读取节点标签表: {node_labels_path}")
    df = pd.read_csv(node_labels_path)

    if "Node" not in df.columns:
        raise ValueError("Node_Labels.csv 中未找到 'Node' 列，请检查文件格式。")

    print(f"[INFO] 载入 SGD 必需基因列表: {sgd_list_path}")
    sgd_set = load_orf_set(sgd_list_path)

    print(f"[INFO] 载入 OGEE 必需基因列表: {ogee_list_path}")
    ogee_set = load_orf_set(ogee_list_path)

    print(f"[INFO] 载入 DEG 必需基因列表: {deg_list_path}")
    deg_set = load_orf_set(deg_list_path)

    # 逐列标记
    nodes = df["Node"].astype(str)

    df["SGD"]  = nodes.isin(sgd_set).astype(int)
    df["OGEE"] = nodes.isin(ogee_set).astype(int)
    df["DEG"]  = nodes.isin(deg_set).astype(int)

    # 计算 SOD 和 essential
    df["SOD"] = df[["SGD", "OGEE", "DEG"]].sum(axis=1)
    df["essential"] = (df["SOD"] >= 1).astype(int)

    print(f"[INFO] 新增列完成，开始保存到: {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("[DONE] Node_Labels_with_essential.csv 已生成。")

# ---------------- main ----------------
def main():
    # 1) 先从 phenotype_data.tab 生成 SGD_essential_orfs.txt
    if not SGD_LIST_PATH.exists():
        build_sgd_essential_list(PHENO_PATH, SGD_LIST_PATH)
    else:
        print(f"[INFO] 已存在 SGD 列表文件: {SGD_LIST_PATH}，跳过生成步骤。")

    # 2) 基于 SGD/OGEE/DEG 列表给 Node_Labels.csv 加列
    annotate_node_labels(
        NODE_LABELS_PATH,
        SGD_LIST_PATH,
        OGEE_LIST_PATH,
        DEG_LIST_PATH,
        OUTPUT_NODE_LABELS,
    )

    # 3) 把Node_Labels_with_essential.csv复制到根目录下的data下，以备后面的程序使用
    shutil.copy(OUTPUT_NODE_LABELS, ROOT_DATA)

if __name__ == "__main__":
    main()
