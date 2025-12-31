#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a list of essential yeast genes ORFs from DEG's deg_annotation_e.csv and SGD's SGD_features.tab: DEG_essential_orfs.txt

Input files (in the same directory):
- deg_annotation_e.csv
- SGD_features.tab

Output file:
- DEG_essential_orfs.txt (one YxxxxW/C per line)
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
    Read deg_annotation_e.csv (; delimited, no header).
    Records of species Saccharomyces cerevisiae were selected.
    """
    if not deg_path.exists():
        raise FileNotFoundError(f"DEG comment file not found: {deg_path}")

    print(f"[INFO] Read DEG comments: {deg_path}")
    # The file is semicolon-separated, enclosed in quotes, and has no header.
    df = pd.read_csv(deg_path, sep=";", header=None, dtype=str)

    # Column 8 (subscript 7) contains the species name, such as  'Saccharomyces cerevisiae'
    mask_sc = df[7].str.contains("Saccharomyces cerevisiae", case=False, na=False)
    df_sc = df[mask_sc].copy()

    # Column 3 (subscript 2) contains gene symbols (such as TFC3, EFB1, etc.).
    df_sc["symbol"] = df_sc[2].astype(str).str.strip()

    print(f"[INFO] Number of Saccharomyces cerevisiae records in DEG: {len(df_sc)}")
    print(f"[INFO] Number of unique symbols: {df_sc['symbol'].nunique()}")
    return df_sc


def load_sgd_feature_map(sgd_path: Path):
    """
    Construct the mapping from SGD_features.tab:
    - standard_name -> systematic_name (YxxxxW/C)
    - The set of ORFs themselves, used to identify symbols that are already ORFs
    """
    if not sgd_path.exists():
        raise FileNotFoundError(f"SGD_features.tab not found: {sgd_path}")

    print(f"[INFO] Read SGD_features: {sgd_path}")
    # The official format is tab-separated, with no header, and a total of 16 columns.
    sgd = pd.read_csv(sgd_path, sep="\t", header=None, dtype=str)

    # Only retain rows with feature type ORF
    sgd_orf = sgd[sgd[1] == "ORF"].copy()

    # Column 4 (subscript 3) is the systematic name YxxxxW/C
    # Column 5 (subscript 4) contains the standard gene name (symbol).
    sgd_orf[3] = sgd_orf[3].astype(str).str.strip()
    sgd_orf[4] = sgd_orf[4].fillna("").astype(str).str.strip()

    # Construct symbol -> ORF mapping
    sym_to_orf = {}
    for _, row in sgd_orf.iterrows():
        sym = row[4]
        if not sym:
            continue
        orf = row[3]
        # If a symbol corresponds to multiple ORFs, we simply retain the first one here.
        if sym not in sym_to_orf:
            sym_to_orf[sym] = orf

    orf_set = set(sgd_orf[3])

    print(f"[INFO] Number of ORFs in SGD: {len(orf_set)}")
    print(f"[INFO] Number of genes with standard names in SGD: {len(sym_to_orf)}")
    return sym_to_orf, orf_set


def build_symbol_to_orf_mapper(sym_to_orf: dict, orf_set: set):
    """
    Returns a function: symbol -> ORF
    Processing logic:
    1. If the symbol contains '/', split it into two and try mapping them separately.
    2. If the symbol itself looks like YGR129W and is in the ORF set, use it directly.
    3. Otherwise, use sym_to_orf to look up the standard name mapping.

    """
    orf_pattern = re.compile(r"Y[A-Z0-9]{2}[0-9]{3}[WC](-[A-Z])?$")

    def symbol_to_orf(symbol: str):
        if symbol is None:
            return None
        s = symbol.strip()
        if not s:
            return None

        # Case 1: For TIM12/MRS5, try once on each side.
        if "/" in s:
            for part in s.split("/"):
                p = part.strip()
                if p in sym_to_orf:
                    return sym_to_orf[p]
            # If no match is found after splitting the parts, proceed with the following logic.

        # Scenario 2: It is inherently in ORF format, such as YGR129W
        if orf_pattern.match(s) and s in orf_set:
            return s

        # Case 3: Standard name, such as TFC3
        return sym_to_orf.get(s)

    return symbol_to_orf


def main():
    # 1) Load the yeast record from DEG
    deg_sc = load_deg_for_yeast(DEG_ANN_PATH)

    # 2) Build the mapping from SGD_features.tab
    sym_to_orf, orf_set = load_sgd_feature_map(SGD_FEAT_PATH)
    symbol_to_orf = build_symbol_to_orf_mapper(sym_to_orf, orf_set)

    # 3) Perform symbol -> ORF mapping
    print("[INFO] Start doing symbol -> ORF mapping...")
    deg_sc["ORF"] = deg_sc["symbol"].apply(symbol_to_orf)

    mapped = deg_sc[deg_sc["ORF"].notna()].copy()
    unmapped = deg_sc[deg_sc["ORF"].isna()].copy()

    print(f"[INFO] Number of successfully mapped records: {len(mapped)}")
    print(f"[INFO] Number of mapping failure records: {len(unmapped)}")

    if len(unmapped) > 0:
        print("[WARN] The following symbols did not match any ORFs (you can manually search again):")
        print(unmapped["symbol"].unique()[:20])

    # 4) Extract the unique ORF, sort it, and write it out.
    orfs = (
        mapped["ORF"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    orfs.to_csv(OUT_PATH, index=False, header=False, encoding="utf-8")
    print(f"[DONE] DEG_essential_orfs.txt has been generated, number of ORFs: {len(orfs)}")
    print(f"[PATH] {OUT_PATH}")


if __name__ == "__main__":
    main()
