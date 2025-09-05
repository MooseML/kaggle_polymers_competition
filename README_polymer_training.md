# Polymer Property Prediction Pipeline

## Overview

This pipeline trains a GNN+Transformer hybrid model to predict 5 polymer properties (`Tg`, `FFV`, `Tc`, `Density`, `Rg`) from SMILES strings using a frozen pre-trained backbone from HOMO-LUMO gap prediction.

## Fixed Issues

✅ **Fixed FileNotFoundError**: Removed incorrect PCQM4M imports that were causing crashes in Kaggle  
✅ **Added forward_backbone_only method**: Fixed missing method in `hybrid_backbone.py`  
✅ **Fixed import conflicts**: Created `dataset_polymer_fixed.py` to avoid circular dependencies  
✅ **Weighted MAE computation**: Implemented proper validation metrics  
✅ **CLI training script**: Added `train_polymer.py` with full configuration options  

## Files Structure

```
kaggle_polymers_competition/
├── build_polymer_lmdb.py       # Build LMDBs from polymer CSV data
├── dataset_polymer_fixed.py    # Fixed dataset classes (no PCQM4M conflicts)
├── graphdata_lmdb.py           # LMDB utilities and GraphData class
├── hybrid_backbone.py          # GNN+Transformer backbone (with forward_backbone_only)
├── polymer_model.py            # PolymerPredictor wrapper model
├── polymers.ipynb              # Fixed notebook for Kaggle
├── train_polymer.py            # Complete training script with CLI
└── README_polymer_training.md  # This file
```

## Usage

### Option 1: Command Line Training Script

```bash
# Basic training (frozen backbone)
python train_polymer.py

# With custom parameters
python train_polymer.py --epochs 50 --lr 2e-4 --batch_size 32

# Unfreeze backbone for fine-tuning
python train_polymer.py --unfreeze --lr 5e-5 --epochs 20

# Include HOMO-LUMO gap head
python train_polymer.py --use_gap --save_model

# Full parameter list
python train_polymer.py --help
```

### Option 2: Kaggle Notebook

Use the fixed notebook `polymers.ipynb` which:
1. Builds LMDBs automatically
2. Trains with proper validation split
3. Generates submission.csv

## Training Pipeline Steps

### 1. Data Preparation (Kaggle Environment)

```bash
# Build training and test LMDBs
python build_polymer_lmdb.py train
python build_polymer_lmdb.py test
```

### 2. Model Training

```python
from dataset_polymer_fixed import LMDBDataset
from polymer_model import PolymerPredictor

# Initialize model with frozen backbone
model = PolymerPredictor(
    backbone_ckpt='best_gnn_transformer_hybrid.pt',
    freeze=True,
    use_gap=False
)

# Train with weighted MAE loss
# ... (see train_polymer.py for full implementation)
```

### 3. Validation & Metrics

The pipeline computes:
- Per-property MAE for each of the 5 targets
- Overall weighted MAE across all valid predictions
- Handles missing labels (NaN values) correctly

### 4. Submission Generation

Generates `submission.csv` with format:
```csv
id,Tg,FFV,Tc,Density,Rg
1109053969,0.1,0.2,0.3,0.4,0.5
...
```

## Key Features

### Architecture
- **Backbone**: Pre-trained GNN+Transformer from HOMO-LUMO gap prediction
- **Head**: New 5-output MLP for polymer properties
- **Freezing**: Backbone can be frozen or fine-tuned

### Training
- **OneCycle LR**: Peak learning rate scheduling
- **Weighted MAE**: Handles missing labels properly
- **Gradient Clipping**: For training stability
- **Stratified Split**: Based on label availability

### Data Handling
- **LMDB Storage**: Fast data loading
- **3D Conformers**: ETKDG geometry generation
- **RBF Encoding**: Distance-based edge features
- **Hop Distance**: Pre-computed attention bias

## Command Line Options

```
--epochs INT        Number of training epochs (default: 30)
--batch_size INT    Batch size (default: 64)
--lr FLOAT          Peak learning rate for OneCycle (default: 1e-4)
--unfreeze          Unfreeze backbone for fine-tuning (default: frozen)
--use_gap           Include HOMO-LUMO gap prediction head (default: False)
--val_split FLOAT   Validation split ratio (default: 0.1)
--seed INT          Random seed (default: 42)
--device STR        Device to use: cuda/cpu/auto (default: auto)
--save_model        Save the best model checkpoint
```

## Expected Performance

With the frozen backbone approach:
- **Training time**: ~30 epochs on GPU
- **Validation MAE**: Should improve over naive baselines
- **Submission**: Ready for Kaggle upload

## Troubleshooting

### Import Errors
- Use `dataset_polymer_fixed.py` instead of `dataset_polymer.py`
- Make sure all required packages are installed in Kaggle environment

### LMDB Issues
- **Fixed**: LMDBs now write to `/kaggle/working/processed_chunks` (writable) instead of `/kaggle/input/polymer/processed_chunks` (read-only)
- Ensure LMDBs are built before training
- Check file paths in Kaggle vs local environment

### Path Errors in Kaggle
- **Fixed**: Notebook now automatically detects Kaggle environment and uses correct paths
- Data reads from `/kaggle/input/neurips-open-polymer-prediction-2025/`
- LMDBs write to `/kaggle/working/processed_chunks/`
- Backbone loads from `/kaggle/input/polymer/best_gnn_transformer_hybrid.pt`

### Memory Issues
- Reduce batch size
- Use fewer workers in DataLoader

### Performance Issues
- Try unfreezing backbone with lower learning rate
- Experiment with the gap head option
- Adjust validation split ratio

## Development Notes

The key insight is using the pre-trained HOMO-LUMO gap prediction backbone as a strong feature extractor for polymer properties. The `forward_backbone_only` method extracts the 512-dimensional CLS token representation, which is then fed to a task-specific head.

The weighted MAE loss properly handles the sparsity in polymer property labels, ensuring stable training even when many target values are missing. 