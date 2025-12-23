#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 OGEE v3 下载的 Saccharomyces cerevisiae W303_genes.csv
和 SGD 的 SGD_features.tab 中，生成 OGEE_essential_orfs.txt

输入文件（放在同一目录，或自己改路径）:
- Saccharomyces cerevisiae W303_genes.csv
- SGD_features.tab

输出文件:
- OGEE_essential_orfs.txt  # 每行一个 ORF，例如 YGR129W
"""

from pathlib import Path
import pandas as pd


# ===== 路径设置（按需修改） =====
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

OGEE_CSV_PATH = DATA_DIR / "Saccharomyces cerevisiae W303_genes.csv"
SGD_FEATURES_PATH = DATA_DIR / "SGD_features.tab"

OUT_OGEE_ORFS = OUTPUT_DIR / "OGEE_essential_orfs.txt"


def build_sgdid_to_orf_map(features_path: Path) -> dict:
    """
    从 SGD_features.tab 构建:
        SGDID (S000000001) -> ORF (YAL001C)
    只使用 feature_type == 'ORF' 的行
    """
    if not features_path.exists():
        raise FileNotFoundError(f"找不到 SGD_features.tab: {features_path}")

    print(f"[INFO] 读取 SGD_features.tab: {features_path}")
    # 文件是 tab 分隔、无表头
    df = pd.read_csv(features_path, sep="\t", header=None, dtype=str)

    # 按 SGD 官方说明:
    #  0: SGDID
    #  1: feature_type
    #  3: feature_name (systematic name, 例如 YGR129W)
    df_orf = df[df[1] == "ORF"].copy()

    sgdid_to_orf = (
        df_orf[[0, 3]]
        .dropna()
        .drop_duplicates(subset=0)
        .set_index(0)[3]
        .to_dict()
    )

    print(f"[INFO] 映射表构建完成, ORF 条目数: {len(sgdid_to_orf)}")
    return sgdid_to_orf


def main():
    # 1. 构建 SGDID -> ORF 映射
    sgd_map = build_sgdid_to_orf_map(SGD_FEATURES_PATH)

    # 2. 读取 OGEE W303 CSV
    if not OGEE_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到 OGEE 基因文件: {OGEE_CSV_PATH}")

    print(f"[INFO] 读取 OGEE W303 基因文件: {OGEE_CSV_PATH}")
    df = pd.read_csv(OGEE_CSV_PATH, dtype=str)

    # 看一下列名（调试用）
    print("[INFO] OGEE 列名:", list(df.columns))

    # 典型列：
    # ['dataset', 'taxaID', 'locus', 'gene', 'essentiality', 'pmid', 'Ref_db']
    # essentiality 列里常见值：'E' (essential), 'NE' (non-essential), 'C' (conditional) 等

    # 3. 只保留 essentiality 标记为 essential 的基因
    ess = df["essentiality"].fillna("").str.upper()
    mask_essential = ess.isin(["E", "ESSENTIAL", "ES"])  # 视具体文件而定，这里兼容几种写法

    df_ess = df[mask_essential].copy()
    print(f"[INFO] OGEE 中 essential 基因行数: {len(df_ess)}")

    # 4. 用 locus (S000000001) 映射到 ORF
    loci = df_ess["locus"].fillna("")

    orfs = []
    missing = []

    for sgdid in loci:
        orf = sgd_map.get(sgdid)
        if orf:
            orfs.append(orf)
        else:
            missing.append(sgdid)

    print(f"[INFO] 成功映射到 ORF 的数量: {len(orfs)}")
    if missing:
        print(f"[WARN] 有 {len(missing)} 个 SGDID 在 SGD_features.tab 中找不到 ORF，示例: {missing[:5]}")

    # 5. 去重、排序并保存为一列文本
    ser_orf = (
        pd.Series(orfs, name="ORF")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    ser_orf.to_csv(OUT_OGEE_ORFS, index=False, header=False, encoding="utf-8")
    print(f"[DONE] 已生成 OGEE_essential_orfs.txt: {OUT_OGEE_ORFS}")
    print(f"[INFO] 最终 ORF 数量: {len(ser_orf)}")


if __name__ == "__main__":
    main()
