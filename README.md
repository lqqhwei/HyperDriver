# Key Driver Protein Identification Based on Spectral Energy Proxy and Multi-scale Spatio-temporal Hypergraph
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18102406.svg)](https://doi.org/10.5281/zenodo.18102406)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Submission-green.svg)](#citation)


## Overview
In the research of dynamic Protein-Protein Interaction (PPI) networks, identifying key driver proteins with low energy consumption is crucial for revealing the evolutionary principles of living systems. Traditional methods predominantly rely on static topological metrics to identify structural hubs. However, due to the neglect of temporal system characteristics and dynamic controllability, these methods often encounter bottlenecks associated with excessive control energy consumption. To address this, this study proposes the HyperDriver computational framework. This framework first leverages knowledge distillation and multi-scale spatio-temporal hypergraph techniques to accurately reconstruct dynamic weights from sparse data and capture high-order synergistic information. It then introduces a spectral energy proxy metric, integrated with an iterated greedy algorithm and adaptive K-Means clustering, to precisely pinpoint optimal driver proteins from the natural discontinuities in data distribution, thereby effectively circumventing the subjective bias of empirical thresholds. Experiments on 12 real-world yeast dynamic datasets demonstrate that the nodes identified by HyperDriver reduce the control energy required for system state transitions by 2–3 orders of magnitude. The study further reveals the intrinsic decoupling mechanism between structural hubs and control hubs, quantitatively validates the critical role of dynamic temporal features and high-order synergistic information in minimizing control energy, and provides a rigorous computational paradigm for identifying precise regulatory targets with low control energy demands.


## Schematic diagram of the overall HyperDriver computational framework
This figure illustrates the five core modules—knowledge distillation for dynamic modeling, multi-scale spatio-temporal hypergraph, spectral energy proxy metric, iterated greedy algorithm, and adaptive K-Means clustering—designed to systematically identify the complete technical path for low-energy-consumption key driver proteins.

![Framework Overview](docs/Figure1.png)


## Project Structure
```text
HyperDriver/                      # Thesis Reproduction Project Root Directory
├── conf/                         # Configuration files for datasets
├── data/                         # Raw datasets (e.g., Static_PPIN, Dynamic_PPIN, Node_Features)
├── docs/                         # Required images and file directory for readme
├── figures/                      # All experimental images generated after the main script is executed
├── results/                      # All dataset CSV result files generated after the main script is executed
├── checkpoints/                  # Stores trained model weights and configurations for full model evaluation
├── processed/                    # Holds cleaned, unified, and index-mapped protein datasets for training
├── casestudies/                     # Case Directory
│   ├── driver_case_study/        # Functional subgraphs and physical verification
│       ├── output/               # Results output directory
│       └── src/                  # Case source code directory
│           └── main.py           # [CASE SCRIPT] One-click script execution
│   ├── energy_case_study/        # Driving patterns and biological mechanisms
│       ├── output                # Results output directory
│       └── src/                  # Case source code directory
│           └── main.py           # [CASE SCRIPT] One-click script execution
├── src/                          # Core source code
│   ├── control_engine.py         # Spectral energy proxy & greedy search
│   ├── data_utils.py             # This script cleans and unifies protein features, labels, and networks
│   ├── hyper_driver.py           # Model + scoring pipeline
│   └── layers.py                 # Basic neural network / HGNN layers
├── baselines_centrality.py       # Baselines: DC / BC / EC
├── eval_driver.py                # Main evaluation logic
├── plot_nature_figs.py           # Visualization suite for paper figures
├── preprocess_loop.py            # This script automates preprocessing for all enabled datasets in the configuration
├── train_hyperdriver.py          # This script trains the HyperDriver model using node features and networks
├── main.py                       # [MASTER SCRIPT] The main experimental execution script
├── LICENSE                       # Defines legal permissions and restrictions for using or sharing code
├── .gitignore                    # Lists files or folders for Git to intentionally exclude
└── README.md                     # Readme Detailed Reading Document
```


## Prerequisites
- **Python**: 3.12+
- **PyTorch**: 2.5+
- OS: Windows / Linux / macOS (tested on Windows 10 + NVIDIA GeForce RTX 4060 Laptop GPU + 13th Gen Intel(R) Core(TM) i9-13900H 2.60GHz + 16GB MEM)


## Installation
### Step 1: Clone the repository
```bash
git clone https://github.com/lqqhwei/HyperDriver.git
```

### Step 2: Install dependencies
```bash
pip install -r docs/requirements.txt
```
> If you use CUDA, please install a PyTorch build that matches your CUDA version first.


## Reproduction Instructions
### Step 1: Main Experiment Pipeline (One-Click)

Run the master script from the repository root to execute the full pipeline:

- **Script location:** `main.py`
- **What it does:**
  1. Control Energy and Performance Comparison(Experiment).
  2. Module Contribution Analysis and Ablation Study(Experiment).
  3. Decoupling Topological Features from Control Energy(Experiment).

- **Command:**
```bash
# Ensure you are in the root directory
python main.py
```

- **Outputs:**
  - Results are saved under `results/<dataset>` / `baselines`,`full`,`keys`.
  - Figures are saved under `figures` / `ablation_battles`,`energy_battles`,`global_summary`,`top_drivers`.

### Step 2: Functional Subgraphs and Physical Verification(Case Study)
- **Script location:** `casestudies/energy_case_study/main.py`
- **Command:**

```bash
# cd casestudies/energy_case_study
python main.py
```

- **Output:** generates `casestudies\energy_case_study\output`.

### Step 3: Driving Patterns and Biological Mechanisms(Case Study)
- **Script location:** `casestudies/driver_case_study/main.py`
- **Command:**

```bash
# cd casestudies/driver_case_study
python main.py
```

- **Output:** generates `casestudies\driver_case_study\output`.


## Datasets
The dynamic yeast DPPIN datasets used in this project are obtained from the original **DPPIN** repository.

- **Source (original release):** https://github.com/DongqiFu/DPPIN
- **Description:** 12 dynamic yeast networks integrating static PPI edges with time-course gene expression.
- **Format:** Raw files (e.g., `Static_PPIN.txt`, `Dynamic_PPIN.txt`, `Node_Features.txt`) are organized under `data/<DatasetName>/`.

If you use the DPPIN datasets, please cite the original paper:

```bibtex
@inproceedings{DBLP:conf/bigdataconf/FuH22,
  author    = {Dongqi Fu and
               Jingrui He},
  title     = {{DPPIN:} {A} Biological Repository of Dynamic Protein-Protein Interaction
               Network Data},
  booktitle = {{IEEE} International Conference on Big Data, Big Data 2022, Osaka,
               Japan, December 17-20, 2022},
  pages     = {5269--5277},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/BigData55660.2022.10020904},
  doi       = {10.1109/BigData55660.2022.10020904}
}
```
We thank the authors of DPPIN for making the datasets publicly available.


## Citation
If you find this code useful, please cite our paper:

```bibtex
@article{HyperDriver2025,
  title   = {Key Driver Protein Identification Based on Spectral Energy Proxy and Multi-scale Spatio-temporal Hypergraph},
  author  = {Qiangqiang Li},
  note    = {Manuscript submitted, under review},
  year    = {2026}
}
```

## Contact
For questions, please open a GitHub issue in this repository.

---
## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
