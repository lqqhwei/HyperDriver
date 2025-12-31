#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
From the Saccharomyces cerevisiae W303_genes.csv file downloaded from OGEE v3
and the SGD_features.tab file from SGD, generate OGEE_essential_orfs.txt

Input files (place in the same directory, or change the path):
- Saccharomyces cerevisiae W303_genes.csv
- SGD_features.tab

Output file:
- OGEE_essential_orfs.txt # One ORF per line, e.g., YGR129W
"""

from pathlib import Path
import pandas as pd


# ===== Path settings (modify as needed) =====
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

OGEE_CSV_PATH = DATA_DIR / "Saccharomyces cerevisiae W303_genes.csv"
SGD_FEATURES_PATH = DATA_DIR / "SGD_features.tab"

OUT_OGEE_ORFS = OUTPUT_DIR / "OGEE_essential_orfs.txt"


def build_sgdid_to_orf_map(features_path: Path) -> dict:
    """
    Build from SGD_features.tab:
        SGDID (S000000001) -> ORF (YAL001C)
    Only use lines where feature_type == 'ORF'
    """
    if not features_path.exists():
        raise FileNotFoundError(f"SGD_features.tab not found: {features_path}")

    print(f"[INFO] Read SGD_features.tab: {features_path}")
    # The file is tab-separated and has no header.
    df = pd.read_csv(features_path, sep="\t", header=None, dtype=str)

    # According to the official SGD statement:
    #  0: SGDID
    #  1: feature_type
    #  3: feature_name (systematic name, For example, YGR129W)
    df_orf = df[df[1] == "ORF"].copy()

    sgdid_to_orf = (
        df_orf[[0, 3]]
        .dropna()
        .drop_duplicates(subset=0)
        .set_index(0)[3]
        .to_dict()
    )

    print(f"[INFO] Mapping table construction complete, ORF entry count: {len(sgdid_to_orf)}")
    return sgdid_to_orf


def main():
    # 1. Construct SGDID -> ORF mapping
    sgd_map = build_sgdid_to_orf_map(SGD_FEATURES_PATH)

    # 2. Read OGEE W303 CSV
    if not OGEE_CSV_PATH.exists():
        raise FileNotFoundError(f"OGEE gene file not found: {OGEE_CSV_PATH}")

    print(f"[INFO] Reading the OGEE W303 gene file: {OGEE_CSV_PATH}")
    df = pd.read_csv(OGEE_CSV_PATH, dtype=str)

    # Take a look at the column names (for debugging purposes).
    print("[INFO] OGEE column names:", list(df.columns))

    # Typical columns:
    # ['dataset', 'taxaID', 'locus', 'gene', 'essentiality', 'pmid', 'Ref_db']
    # Common values ​​for the essentiality column include: 'E' (essential), 'NE' (non-essential), 'C' (conditional), etc.

    # 3. Only genes marked as essentiality are retained.
    ess = df["essentiality"].fillna("").str.upper()
    mask_essential = ess.isin(["E", "ESSENTIAL", "ES"])  # Depending on the specific document, several writing styles are supported here.

    df_ess = df[mask_essential].copy()
    print(f"[INFO] Number of essential gene rows in OGEE: {len(df_ess)}")

    # 4. Map locus(S000000001) to ORF
    loci = df_ess["locus"].fillna("")

    orfs = []
    missing = []

    for sgdid in loci:
        orf = sgd_map.get(sgdid)
        if orf:
            orfs.append(orf)
        else:
            missing.append(sgdid)

    print(f"[INFO] Number of successfully mapped to ORF: {len(orfs)}")
    if missing:
        print(f"[WARN] There are {len(missing)} SGDIDs whose ORFs cannot be found in SGD_features.tab. Example: {missing[:5]}")

    # 5. Remove duplicates, sort, and save as a column of text.
    ser_orf = (
        pd.Series(orfs, name="ORF")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    ser_orf.to_csv(OUT_OGEE_ORFS, index=False, header=False, encoding="utf-8")
    print(f"[DONE] OGEE_essential_orfs.txt has been generated: {OUT_OGEE_ORFS}")
    print(f"[INFO] Final ORF count: {len(ser_orf)}")


if __name__ == "__main__":
    main()
