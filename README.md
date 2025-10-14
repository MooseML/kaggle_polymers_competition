# Polymer Property Prediction (NeurIPS Open Polymer 2025)

**Bronze medal (top ~6%) — 0.088 wMAE (private LB)**
Predicting five polymer properties directly from SMILES using a **hybrid graph learning + tabular ensemble** pipeline.

<p align="center">
  <b>Targets:</b> Tg (glass transition), FFV (fractional free volume), Tc (thermal conductivity), Density, Rg (radius of gyration)
</p>

---

## TLDR

* **Hybrid modeling:** Graph Transformer / GNN backbones + **RDKit global descriptors** + **3D conformers**.
* **Diverse ensemble:** XGBoost / LightGBM / ExtraTrees + neural models, blended **per property** using cross-validated predictions.
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
* [Ensembling](#ensembling)
* [Results](#results)
* [Notes & Best Practices](#notes--best-practices)

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

## Setup

```bash
# Python >=3.10 recommended
python -m venv .venv && source .venv/bin/activate

pip install torch torch-geometric
pip install ogb rdkit-pypi
pip install scikit-learn lightgbm xgboost
pip install optuna pandas numpy polars tqdm
pip install matplotlib
```

> If you use GPUs, install the CUDA-matched PyTorch wheel first.

---

## Data

1. Place `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`.
2. (Optional) Put any **supplemental** sources you want to merge (identical canonical SMILES) under `data/supplements/`.
3. Build LMDB graph shards:

```bash
python scripts/data_preprocessing/build_lmdb.py train
python scripts/data_preprocessing/build_lmdb.py test
```

---

## Quickstart

### Train a main model

```bash
python scripts/model_training/train_polymer.py \
  --backbone graph_transformer \
  --data_root data/processed_chunks \
  --cv_folds 5 --seed 42 \
  --use_rdkit_globals 1 --use_3d 1
```

### Produce a submission

```bash
python scripts/model_training/train_polymer.py \
  --inference_only 1 \
  --checkpoint_dir saved_models/graph_transformer_fold* \
  --write_submission submissions/gt_submission.csv
```

---

## Models

### Graph Neural Networks (GNN)

* **GINEConv** backbone with bond/edge encoders and distance-aware features.
* **Task heads per property** (shared trunk, property-specific MLPs).
* **Hybrid late fusion**: concatenates pooled graph embedding with **global RDKit descriptors** (`u`) before the head.

### Graph Transformer

* GPS-style attention with shortest-path/structural biases (long-range interactions).
* Same **hybrid fusion** option with global descriptors.
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
* **Metrics:** MAE / RMSE / R² per property + competition **wMAE**.

---

## Ensembling

**Key idea:** blend, don’t “pick the winner”. You get variance reduction and complementary error patterns across model families.

1. **Generate OOF & test predictions** for each base model (per property).
2. **Per-label blending**:

   * Start with a **simple weighted average**: `0.6 * GBM + 0.4 * GNN/GT`
   * Or learn weights with **ridge/ElasticNet** on OOF (stacking).
3. **Fold averaging**: average predictions across CV folds for each base learner.
4. **Final submission**: average/blend per property, then stitch the 5 tasks into the submission CSV.

Example (simple blend):

```python
# pseudo-code sketch
blend = 0.6 * xgb_pred + 0.4 * gnn_pred
submission = pd.concat([blend_Tg, blend_FFV, blend_Tc, blend_Density, blend_Rg], axis=1)
submission.to_csv("submissions/ensemble.csv", index=False)
```

---

## Results

### Leaderboard

* **Private LB:** **0.088 wMAE** → **Bronze medal (~top 6%)**

### Per-property (validation)

| Property | Best Model        | MAE       | R²     |
| -------- | ----------------- | --------- | ------ |
| FFV      | Graph Transformer | 0.005713  | 0.9223 |
| Tg       | GNN2              | 47.105114 | 0.6040 |
| Tc       | GNN2_Aug          | 0.025252  | 0.8157 |
| Rg       | ExtraTrees_Aug    | 1.609396  | 0.7227 |
| Density  | ExtraTrees_Aug    | 0.028135  | 0.8850 |

**Insights**

* **Graph Transformers** shine on FFV (packing/long-range cues).
* **GNNs** lead Tg/Tc (thermal properties).
* **Tree-based** models are robust on Rg/Density when fed curated descriptors.
* **3D features** help across the board.
* **Blending** beats per-property “winner selection”.

---

## Notes & Best Practices

* **Use both worlds**: graph learning + descriptor-rich boosters.
* **Property-specific configs**: heads, regularization, feature trims.
* **Keep OOFs**: required for reliable stacking/weight learning.
* **Determinism**: fix seeds; log folds, versions, and feature sets.
* **Small changes, big gains**: fold averaging, RobustScaler(y), gradient clipping.

---