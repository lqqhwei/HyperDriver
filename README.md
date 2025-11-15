# HyperDriver

A minimal and clean repository structure for the **HyperDriver** framework — a dynamic PPIN modeling and hypergraph-based driver protein identification pipeline.

## Directory Structure
```
D:/SCI
├── conf/               # Dataset configuration files
├── data/               # Raw PPIN, expression, and label data
├── outputs/            # Single-run outputs
├── outputs_multi/      # Multi-dataset outputs
└── scripts/            # All processing, modeling, and ablation scripts
```

## How to Run
1. Prepare data under `data/` and config files under `conf/`.
2. Execute scripts in order:
   - `01_preprocess.py`
   - `02_dynamic_graph.py`
   - `03_hgnn.py`
   - `04_drivers_hgnn.py`
   - `05_eval_hgnn.py`
   - (Optional) ablation scripts `07`–`12`
3. Multi-dataset execution:
   ```
   python scripts/15_run_multi_datasets.py
   ```

## Requirements
- Python 3.x
- Common scientific libraries (NumPy, SciPy, PyTorch, etc.)

## License
This project is licensed under the MIT License.
