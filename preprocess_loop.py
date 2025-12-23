# preprocess_loop.py

import os

from src.data_utils import (
    load_datasets_config,
    preprocess_single_dataset,
)


def main():
    # 当前脚本所在目录就是项目根目录 D:/HYPERDRIVER
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 读取数据集配置 :contentReference[oaicite:13]{index=13}
    conf_path = os.path.join(root_dir, "conf", "datasets.json")
    enabled_datasets = load_datasets_config(conf_path)

    print("========== HyperDriver Preprocess ==========")
    print(f"[INFO] ROOT_DIR = {root_dir}")
    print(f"[INFO] datasets.json = {conf_path}")
    print(f"[INFO] Enabled datasets ({len(enabled_datasets)}):")
    for name in enabled_datasets:
        print("  -", name)

    # 2. 对每个启用数据集做预处理
    for name in enabled_datasets:
        print("\n--------------------------------------------")
        print(f"[INFO] Preprocessing dataset: {name}")
        try:
            preprocess_single_dataset(root_dir, name)
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {name}: {e}")


if __name__ == "__main__":
    main()
