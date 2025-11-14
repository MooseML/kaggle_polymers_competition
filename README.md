# Polymer Property Prediction [NeurIPS - Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)


**Bronze medal (top ~6%) - 0.088 wMAE (private LB)**
Predicting five polymer properties directly from SMILES using a hybrid graph learning + tabular ensemble pipeline. 
<p align="center">
  <b>Targets:</b> Tg (glass transition), FFV (fractional free volume), Tc (thermal conductivity), Density, Rg (radius of gyration)
</p>

---

## TLDR

* **Hybrid modeling:** Graph Transformer / GNN backbones + RDKit global descriptors + 3D conformers.
* **Diverse ensemble:** XGBoost / LightGBM / ExtraTrees + neural models, blended per property using cross-validated predictions.
* **Reproducible pipeline:** LMDB graph storage, deterministic CV, weighted loss for missing labels, Optuna tuning.

---

## Table of Contents

* [Repo Structure](#repo-structure)
* [Setup](#setup)
* [Data](#data)
* [Quickstart](#quickstart)
* [Models](#models)
* [Feature Engineering](#feature-engineering)
* [Training & Evaluation](#training--evaluation)
* [Results](#results)

---

## Repo Structure

```
kaggle_polymers_competition/
├── data/
│   ├── train.csv               # SMILES + targets
│   ├── test.csv                # SMILES only
│   ├── sample_submission.csv
│   ├── homolumo_parent.csv     # Optional electronic features
│   ├── processed_chunks/       # LMDB shards
│   └── supplements/            # External/canonicalized extras (optional)
├── notebooks/
│   ├── 01_exploratory/         # EDA & single-task probes
│   ├── 02_baselines/           # RF/ET/GBM baselines
│   ├── 03_main_models/         # GNN, Graph Transformer, hybrid runs
│   └── 04_post_competition/    # Post-hoc analysis
├── src/
│   ├── data/                   # Graph building, LMDB, datasets
│   └── models/                 # GNN/GT backbones, heads, fusion blocks
├── scripts/
│   ├── data_preprocessing/
│   │   └── build_lmdb.py
│   └── model_training/
│       └── train_polymer.py
├── saved_models/
└── submissions/
```

---

## Local Environment Setup (GPU)

This solution was ran **locally** on a GPU system with the following specs:

```
Python 3.8.20
CUDA 11.8
NVIDIA RTX 3070 Ti
PyTorch 2.4.1 + cu118
```

## How to Run
### 1. Clone the Repository
```bash
git clone git@github.com:MooseML/kaggle_polymers_competition.git
cd kaggle_polymers_competition
```

### 2. Set Up the Python Environment

#### With Conda (Recommended for GPU / PyTorch Compatibility)

```bash
conda create --name polymers_comp_env python==3.8.20
conda activate polymers_comp_env
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -f https://data.pyg.org/whl/torch-2.4.1+cu118.html \ torch-geometric==2.6.1 \ torch-scatter==2.1.2+pt24cu118 \ torch-sparse==0.6.18+pt24cu118 \ torch-cluster==1.6.3+pt24cu118

conda env update -n polymers_comp_env -f environment.yml --prune

```

> This setup matches my environment (Python 3.8.20, CUDA 11.8, RTX 3070 Ti, PyTorch 2.4.1).  
> If you're using a different system or Python version, check [PyTorch installation options](https://pytorch.org/get-started/locally/) to match your drivers and hardware.



### 3. Download the Dataset
This dataset can be donwloaded from **Kaggle**:

#### Option 1: Manually Download
1. Go to [Kaggle](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data)
2. Click **Download All**.
3. Extract them into the `data/` directory.

#### Option 2: Run the Kaggle API Script
Ensure your **Kaggle API key (`kaggle.json`)** is set up, then run:
```sh
python scripts/download_data.py
```

## 4. Create LMDBs

1.  Build LMDB graph shards. **Using the direct Python script is generally recommended** for cross-platform consistency and better integration with Conda environments.

### Option 1: Recommended (Python Script) 
Ensure your Conda environment is active and run the script directly:

```bash
python scripts/data_preprocessing/build_lmdb.py train
python scripts/data_preprocessing/build_lmdb.py test
```

### Option 2: Windows Batch File Shortcut 
For convenience on **Windows** systems, you can use the provided batch script. **Note:** This assumes your Python environment is correctly set up and accessible from the command line.

```bash
.\build_lmdb.bat
```

---

## Models

### Graph Neural Networks (GNN)

* **GINEConv** backbone with bond/edge encoders and distance-aware features.
* **Task heads per property** (shared trunk, property-specific MLPs).
* **Hybrid late fusion**: concatenates pooled graph embedding with **global RDKit descriptors** (`u`) before the head.

### Graph Transformer

* GPS-style attention with shortest-path/structural biases (long-range interactions).
* Same hybrid fusion option with global descriptors.
* Positional/geometric encodings when 3D conformers are available.

### Tree-based Models

* **XGBoost / LightGBM / ExtraTrees** with RDKit fingerprints + curated descriptors.
* Per-label hyperparams and feature selection (permutation/importance trims).

---

## Feature Engineering

* **RDKit 2D global descriptors** (MolWt, MolMR, MolLogP, TPSA, FractionCSP3, ring/rotor counts, heteroatom counts, etc.).
* **Fingerprints:** Morgan / MACCS; optional AtomPair/Torsion.
* **3D conformers:** MMFF/UFF-derived geometry stats (distance/angle summaries, planarity/ring features).
* **SMILES canonicalization** for dedup/merge; parent–child aggregation for repeats.
* (Optional) **HOMO–LUMO gap** features (`homolumo_parent.csv`) for electronic cues.

---

## Training & Evaluation

* **Loss:** Weighted MAE per property with masking of missing labels.
* **Optimization:** AdamW + cosine schedule, gradient clipping, mixed precision.
* **Target scaling:** RobustScaler on y for graph models (inverse-transform at inference).
* **CV:** 5-fold stratified regression splits, deterministic seeds, OOF prediction saving.
* **Metrics:** MAE / RMSE / R² per property + competition wMAE.

---


## Results

### Leaderboard

* **Private LB:** **0.088 wMAE** → **Bronze medal (~top 6%)**

### Per-property (validation)

| Property | Best Model        | MAE       | R²     |
| -------- | ----------------- | --------- | ------ |
| FFV      | Graph Transformer | 0.005713  | 0.9223 |
| Tg       | GNN2              | 47.105114 | 0.6040 |
| Tc       | GNN2_Aug          | 0.025115  | 0.8000 |
| Rg       | ExtraTrees_Aug    | 1.532573  | 0.7371 |
| Density  | XGB               | 0.024079  | 0.9339 |

**Insights**

* **Graph Transformers** shine on targets with more data like FFV.
* **GNNs** lead Tg/Tc (thermal properties).
* **Tree-based** models are robust on Rg/Density when fed curated descriptors.
* **3D features** help across the board.
---
