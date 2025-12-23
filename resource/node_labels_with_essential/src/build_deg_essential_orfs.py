#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 DEG 的 deg_annotation_e.csv + SGD 的 SGD_features.tab 中
生成酵母的必需基因 ORF 列表：DEG_essential_orfs.txt

输入文件（放在同一目录）：
  - deg_annotation_e.csv
  - SGD_features.tab

输出文件：
  - DEG_essential_orfs.txt   （每行一个 YxxxxW/C）
"""

from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DEG_ANN_PATH = DATA_DIR / "deg_annotation_e.csv"
SGD_FEAT_PATH = DATA_DIR / "SGD_features.tab"
OUT_PATH      = OUTPUT_DIR / "DEG_essential_orfs.txt"


def load_deg_for_yeast(deg_path: Path) -> pd.DataFrame:
    """
    读取 deg_annotation_e.csv（; 分隔，无表头），
    筛选出物种为 Saccharomyces cerevisiae 的记录。
    """
    if not deg_path.exists():
        raise FileNotFoundError(f"找不到 DEG 注释文件: {deg_path}")

    print(f"[INFO] 读取 DEG 注释: {deg_path}")
    # 文件是分号分隔、带引号、无表头
    df = pd.read_csv(deg_path, sep=";", header=None, dtype=str)

    # 第 8 列（下标 7）是物种名，如 'Saccharomyces cerevisiae'
    mask_sc = df[7].str.contains("Saccharomyces cerevisiae", case=False, na=False)
    df_sc = df[mask_sc].copy()

    # 第 3 列（下标 2）是 gene symbol（比如 TFC3、EFB1 等）
    df_sc["symbol"] = df_sc[2].astype(str).str.strip()

    print(f"[INFO] DEG 中 Saccharomyces cerevisiae 记录数: {len(df_sc)}")
    print(f"[INFO] 不重复 symbol 数: {df_sc['symbol'].nunique()}")
    return df_sc


def load_sgd_feature_map(sgd_path: Path):
    """
    从 SGD_features.tab 构建映射：
      - standard_name -> systematic_name (YxxxxW/C)
      - ORF 本身集合，用于识别已经是 ORF 的 symbol
    """
    if not sgd_path.exists():
        raise FileNotFoundError(f"找不到 SGD_features.tab: {sgd_path}")

    print(f"[INFO] 读取 SGD_features: {sgd_path}")
    # 官方是 tab 分隔、无表头，一共 16 列
    sgd = pd.read_csv(sgd_path, sep="\t", header=None, dtype=str)

    # 只保留特征类型为 ORF 的行
    sgd_orf = sgd[sgd[1] == "ORF"].copy()

    # 第 4 列（下标 3）是系统学名 YxxxxW/C
    # 第 5 列（下标 4）是标准基因名（symbol）
    sgd_orf[3] = sgd_orf[3].astype(str).str.strip()
    sgd_orf[4] = sgd_orf[4].fillna("").astype(str).str.strip()

    # 构建 symbol -> ORF 映射
    sym_to_orf = {}
    for _, row in sgd_orf.iterrows():
        sym = row[4]
        if not sym:
            continue
        orf = row[3]
        # 如果一个 symbol 对应多个 ORF，这里简单保留第一个
        if sym not in sym_to_orf:
            sym_to_orf[sym] = orf

    orf_set = set(sgd_orf[3])

    print(f"[INFO] SGD 中 ORF 数量: {len(orf_set)}")
    print(f"[INFO] SGD 中带标准名的基因数量: {len(sym_to_orf)}")
    return sym_to_orf, orf_set


def build_symbol_to_orf_mapper(sym_to_orf: dict, orf_set: set):
    """
    返回一个函数：symbol -> ORF
    处理逻辑：
      1. 如果 symbol 里有 '/'，拆成两个分别尝试映射
      2. 如果 symbol 本身长得像 YGR129W 这种，并且在 ORF 集合里，就直接用
      3. 否则用 sym_to_orf 查标准名映射
    """
    orf_pattern = re.compile(r"Y[A-Z0-9]{2}[0-9]{3}[WC](-[A-Z])?$")

    def symbol_to_orf(symbol: str):
        if symbol is None:
            return None
        s = symbol.strip()
        if not s:
            return None

        # 情况 1：TIM12/MRS5 这种，用两边各试一次
        if "/" in s:
            for part in s.split("/"):
                p = part.strip()
                if p in sym_to_orf:
                    return sym_to_orf[p]
            # 如果拆开也没匹配到，就先走下面的逻辑

        # 情况 2：本身就是 ORF 格式，比如 YGR129W
        if orf_pattern.match(s) and s in orf_set:
            return s

        # 情况 3：普通标准名，比如 TFC3
        return sym_to_orf.get(s)

    return symbol_to_orf


def main():
    # 1) 载入 DEG 中的酵母记录
    deg_sc = load_deg_for_yeast(DEG_ANN_PATH)

    # 2) 从 SGD_features.tab 构建映射
    sym_to_orf, orf_set = load_sgd_feature_map(SGD_FEAT_PATH)
    symbol_to_orf = build_symbol_to_orf_mapper(sym_to_orf, orf_set)

    # 3) 做 symbol -> ORF 的映射
    print("[INFO] 开始做 symbol -> ORF 映射...")
    deg_sc["ORF"] = deg_sc["symbol"].apply(symbol_to_orf)

    mapped = deg_sc[deg_sc["ORF"].notna()].copy()
    unmapped = deg_sc[deg_sc["ORF"].isna()].copy()

    print(f"[INFO] 映射成功记录数: {len(mapped)}")
    print(f"[INFO] 映射失败记录数: {len(unmapped)}")

    if len(unmapped) > 0:
        print("[WARN] 以下 symbol 暂未匹配到 ORF（可手动再查）：")
        print(unmapped["symbol"].unique()[:20])

    # 4) 提取不重复 ORF，排序后写出
    orfs = (
        mapped["ORF"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    orfs.to_csv(OUT_PATH, index=False, header=False, encoding="utf-8")
    print(f"[DONE] 已生成 DEG_essential_orfs.txt，ORF 数量: {len(orfs)}")
    print(f"[PATH] {OUT_PATH}")


if __name__ == "__main__":
    main()
