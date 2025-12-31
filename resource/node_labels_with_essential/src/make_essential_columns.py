#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a list of essential genes based on the phenotype_data.tab file for SGD,
and add the following 5 columns to Node_Labels.csv:
1) SGD: Whether the ORF is in the inviabl* phenotype list for SGD (0/1)
2) OGEE: Whether it is in the OGEE essential gene list (0/1)
3) DEG: Whether it is in the DEG essential gene list (0/1)
4) SOD: The sum of SGD + OGEE + DEG
5) essential: 1 if SOD >= 1, 0 otherwise

Preparation before use:
- The following files should be in the current directory (or your BASE_DIR):
phenotype_data.tab
Node_Labels.csv
OGEE_essential_orfs.txt
DEG_essential_orfs.txt

Note:
- OGEE_essential_orfs.txt / The DEG_essential_orfs.txt file must be formatted as follows: One ORF per line (e.g., YGR129W), with no header.
"""

from pathlib import Path
import csv
import pandas as pd
import shutil

# ---------------- Path configuration (modify as needed) ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

PHENO_PATH        = DATA_DIR / "phenotype_data.tab"          # The original file downloaded by SGD
NODE_LABELS_PATH  = DATA_DIR / "Node_Labels.csv"             # The table you started with
OGEE_LIST_PATH    = OUTPUT_DIR / "OGEE_essential_orfs.txt"     # You are ready
DEG_LIST_PATH     = OUTPUT_DIR / "DEG_essential_orfs.txt"      # You are ready

SGD_LIST_PATH     = OUTPUT_DIR / "SGD_essential_orfs.txt"      # This script will be generated automatically.
OUTPUT_NODE_LABELS = OUTPUT_DIR / "Node_Labels_with_essential.csv"   # This script will be generated automatically.

ROOT_DIR = Path(__file__).resolve().parents[3]
ROOT_DATA = ROOT_DIR / "data"

# ---------------- Utility functions ----------------
def load_orf_set(txt_path: Path) -> set:
    """
    Load an ORF collection from a text file.
    Requirements: One ORF name per line (e.g., YGR129W). Blank lines are allowed and will be automatically skipped.
    If a line has multiple fields, only the first field will be retrieved.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    orfs = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            orf = line.split()[0]
            orfs.add(orf)
    return orfs

# ---------------- Step 1: Generate SGD_essential_orfs.txt from phenotype_data.tab ----------------
def build_sgd_essential_list(pheno_path: Path, out_path: Path) -> None:
    """
    Filter out all ORFs for the inviabl* phenotype from SGD's phenotype_data.tab.
    Generate a text file named out_path containing a list of ORFs.

    illustrate:
    - Instead of using pandas.read_csv, we will manually parse it using csv.reader.
      Because some rows in the file have 14 columns, while others have 15 or 13 columns.
    - This only relies on three columns:
        0: ORF
        1: feature_type
        9: phenotype observable (containing the word 'inviab')

    Filter out all ORFs for the `inviabl*` phenotype from the `phenotype_data.tab` file in SGD,
    and generate a text file named `out_path` containing the ORFs.

    Notes:
    - Instead of using `pandas.read_csv`, manually parse the data using `csv.reader`,

    because some rows in the file have 14 columns, while others have 15 or 13.
    - Only three columns are used here:
        0: ORF
        1: feature_type
        9: phenotype observable (containing 'inviab')
    """
    if not pheno_path.exists():
        raise FileNotFoundError(f"phenotype_data.tab not found: {pheno_path}")

    print(f"[INFO] Reading phenotype data: {pheno_path}")

    essential_orfs = set()
    total_rows = 0
    used_rows = 0

    with pheno_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            total_rows += 1

            # Skip blank lines or strange lines with too few columns.
            if not row or len(row) < 10:
                continue

            orf = row[0].strip()
            feature_type = row[1].strip().upper()
            phenotype = row[9].strip().lower()  # Column 9 is the phenotype description.

            # As long as ORF type
            if feature_type != "ORF":
                continue

            # Phenotypes containing inviabl are considered necessary.
            if "inviab" in phenotype:
                essential_orfs.add(orf)
                used_rows += 1

    essential_orfs = sorted(essential_orfs)
    print(f"[INFO] Total number of rows: {total_rows}")
    print(f"[INFO] Number of records matching inviabl*: {used_rows}")
    print(f"[INFO] Number of unique ORFs: {len(essential_orfs)}")

    # Save as a text column, one ORF per line.
    with out_path.open("w", encoding="utf-8") as f:
        for orf in essential_orfs:
            f.write(f"{orf}\n")

    print(f"[DONE] A list of genes required for SGD has been generated: {out_path}")

# ---------------- Step 2: Add SGD/OGEE/DEG/SOD/essential to Node_Labels.csv ----------------
def annotate_node_labels(node_labels_path: Path,
                         sgd_list_path: Path,
                         ogee_list_path: Path,
                         deg_list_path: Path,
                         output_path: Path) -> None:
    """
    Read in Node_Labels.csv and add 5 columns based on the 3 list files:
    SGD, OGEE, DEG, SOD, essential
    """
    if not node_labels_path.exists():
        raise FileNotFoundError(f"Node_Labels.csv not found: {node_labels_path}")

    print(f"[INFO] Read the node label table: {node_labels_path}")
    df = pd.read_csv(node_labels_path)

    if "Node" not in df.columns:
        raise ValueError("The 'Node' column was not found in Node_Labels.csv. Please check the file format.")

    print(f"[INFO] Load the list of genes required for SGD: {sgd_list_path}")
    sgd_set = load_orf_set(sgd_list_path)

    print(f"[INFO] Load the list of genes required for OGEE: {ogee_list_path}")
    ogee_set = load_orf_set(ogee_list_path)

    print(f"[INFO] Load the list of genes required for DEG: {deg_list_path}")
    deg_set = load_orf_set(deg_list_path)

    # Column-by-column marking
    nodes = df["Node"].astype(str)

    df["SGD"]  = nodes.isin(sgd_set).astype(int)
    df["OGEE"] = nodes.isin(ogee_set).astype(int)
    df["DEG"]  = nodes.isin(deg_set).astype(int)

    # Calculate SOD and essential
    df["SOD"] = df[["SGD", "OGEE", "DEG"]].sum(axis=1)
    df["essential"] = (df["SOD"] >= 1).astype(int)

    print(f"[INFO] Adding a new column, starting to save to: {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("[DONE] The Node_Labels_with_essential.csv file has been generated.")

# ---------------- main ----------------
def main():
    # 1) First, generate SGD_essential_orfs.txt from phenotype_data.tab.
    if not SGD_LIST_PATH.exists():
        build_sgd_essential_list(PHENO_PATH, SGD_LIST_PATH)
    else:
        print(f"[INFO] An SGD list file already exists: {SGD_LIST_PATH}, skip the generation step.")

    # 2) Add columns to Node_Labels.csv based on the SGD/OGEE/DEG list.
    annotate_node_labels(
        NODE_LABELS_PATH,
        SGD_LIST_PATH,
        OGEE_LIST_PATH,
        DEG_LIST_PATH,
        OUTPUT_NODE_LABELS,
    )

    # 3) Copy Node_Labels_with_essential.csv to the data folder in the root directory for later use in the program.
    shutil.copy(OUTPUT_NODE_LABELS, ROOT_DATA)

if __name__ == "__main__":
    main()
