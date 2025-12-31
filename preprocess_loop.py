# preprocess_loop.py

import os

from src.data_utils import (
    load_datasets_config,
    preprocess_single_dataset,
)


def main():
    # The current script is located in the project root directory D:/HYPERDRIVER
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Read dataset configuration :contentReference[oaicite:13]{index=13}
    conf_path = os.path.join(root_dir, "conf", "datasets.json")
    enabled_datasets = load_datasets_config(conf_path)

    print("========== HyperDriver Preprocess ==========")
    print(f"[INFO] ROOT_DIR = {root_dir}")
    print(f"[INFO] datasets.json = {conf_path}")
    print(f"[INFO] Enabled datasets ({len(enabled_datasets)}):")
    for name in enabled_datasets:
        print("  -", name)

    # 2. Preprocess each enabled dataset
    for name in enabled_datasets:
        print("\n--------------------------------------------")
        print(f"[INFO] Preprocessing dataset: {name}")
        try:
            preprocess_single_dataset(root_dir, name)
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {name}: {e}")


if __name__ == "__main__":
    main()
