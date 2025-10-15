#!/usr/bin/env python3
"""
Polymer Property Prediction Training Script
============================================

Usage:
    python train_polymer.py [options]

Options:
    --epochs INT        Number of training epochs (default: 30)
    --batch_size INT    Batch size (default: 64)
    --lr FLOAT          Peak learning rate for OneCycle (default: 1e-4)
    --unfreeze          Unfreeze backbone for fine-tuning (default: frozen)
    --use_gap           Include HOMO-LUMO gap prediction head (default: False)
    --val_split FLOAT   Validation split ratio (default: 0.1)
    --seed INT          Random seed (default: 42)
    --device STR        Device to use: cuda/cpu/auto (default: auto)
    --save_model        Save the best model checkpoint
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports
from dataset_polymer_fixed import LMDBDataset
from polymer_model import PolymerPredictor

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def weighted_mae_loss(pred, target, mask):
    """
    Compute weighted MAE loss for polymer properties
    
    Args:
        pred: (B, 5) predicted values
        target: (B, 5) target values  
        mask: (B, 5) boolean mask (True where target is not NaN)
    
    Returns:
        Weighted MAE loss (scalar)
    """
    # Only compute loss where we have valid targets
    valid_pred = pred[mask]
    valid_target = target[mask]
    
    if len(valid_pred) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Simple MAE for now - can be enhanced with proper weighting
    mae = F.l1_loss(valid_pred, valid_target)
    return mae

def compute_weighted_mae(pred, target, mask):
    """
    Compute weighted MAE metric for validation
    
    Args:
        pred: (B, 5) predicted values
        target: (B, 5) target values
        mask: (B, 5) boolean mask (True where target is not NaN)
    
    Returns:
        Dictionary with per-property MAE and overall weighted MAE
    """
    label_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    results = {}
    
    for i, col in enumerate(label_cols):
        col_mask = mask[:, i]
        if col_mask.sum() > 0:
            col_mae = F.l1_loss(pred[col_mask, i], target[col_mask, i]).item()
            results[f'{col}_MAE'] = col_mae
        else:
            results[f'{col}_MAE'] = float('nan')
    
    # Overall MAE across all valid predictions
    if mask.sum() > 0:
        overall_mae = F.l1_loss(pred[mask], target[mask]).item()
        results['Overall_MAE'] = overall_mae
    else:
        results['Overall_MAE'] = float('nan')
    
    return results

def create_dataloaders(train_ids, val_ids, train_lmdb_path, batch_size, num_workers=2):
    """Create train and validation dataloaders"""
    train_dataset = LMDBDataset(train_ids, train_lmdb_path)
    val_dataset = LMDBDataset(val_ids, train_lmdb_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_samples = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        pred = model(batch)
        
        # Compute loss mask (where target is not NaN)
        mask = ~torch.isnan(batch.y)
        
        # Compute loss
        loss = weighted_mae_loss(pred, batch.y, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / num_samples

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)
            pred = model(batch)
            
            all_preds.append(pred.cpu())
            all_targets.append(batch.y.cpu())
            all_masks.append(~torch.isnan(batch.y.cpu()))
    
    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Compute metrics
    metrics = compute_weighted_mae(all_preds, all_targets, all_masks)
    
    return metrics

def generate_submission(model, test_lmdb_path, output_path, device):
    """Generate submission file"""
    # Read test data to get IDs
    test_csv = pd.read_csv('data/test.csv')
    test_ids = test_csv['id'].values
    
    # Create test dataset and loader
    test_dataset = LMDBDataset(test_ids, test_lmdb_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Generate predictions
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating predictions'):
            batch = batch.to(device)
            pred = model(batch)
            all_preds.append(pred.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'Tg': all_preds[:, 0],
        'FFV': all_preds[:, 1], 
        'Tc': all_preds[:, 2],
        'Density': all_preds[:, 3],
        'Rg': all_preds[:, 4]
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Polymer Property Predictor')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Peak learning rate')
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze backbone for fine-tuning')
    parser.add_argument('--use_gap', action='store_true', help='Include HOMO-LUMO gap head')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'], help='Device')
    parser.add_argument('--save_model', action='store_true', help='Save best model checkpoint')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Detect environment and set paths
    if os.path.exists('/kaggle'):
        data_root = '/kaggle/input/neurips-open-polymer-prediction-2025'
        chunk_dir = '/kaggle/working/processed_chunks'
        backbone_path = '/kaggle/input/polymer/hlgap-gnn3d-transformer-pcqm4mv2-v1.pt'
    else:
        data_root = 'data'
        chunk_dir = os.path.join(data_root, 'processed_chunks')
        backbone_path = 'hlgap-gnn3d-transformer-pcqm4mv2-v1.pt'
    
    train_lmdb = os.path.join(chunk_dir, 'polymer_train3d_dist.lmdb')
    test_lmdb = os.path.join(chunk_dir, 'polymer_test3d_dist.lmdb')
    
    # Check if LMDBs exist
    if not os.path.exists(train_lmdb):
        print(f"Training LMDB not found at {train_lmdb}")
        print("Please run the LMDB building scripts first:")
        print("  python build_polymer_lmdb.py train")
        print("  python build_polymer_lmdb.py test") 
        sys.exit(1)
    
    # Load training data and create splits
    train_csv_path = os.path.join(data_root, 'train.csv')
    train_csv = pd.read_csv(train_csv_path)
    
    label_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    ids = train_csv['id'].values
    
    # Stratify by number of non-NaN labels per row
    stratify_col = train_csv[label_cols].notna().sum(axis=1)
    train_ids, val_ids = train_test_split(
        ids,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=stratify_col
    )
    
    print(f"Train/val split: {len(train_ids)} / {len(val_ids)} samples")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_ids, val_ids, train_lmdb, args.batch_size
    )
    
    # Initialize model
    model = PolymerPredictor(
        backbone_ckpt=backbone_path,
        freeze=not args.unfreeze,
        use_gap=args.use_gap
    ).to(device)
    
    print(f"Model initialized (backbone {'unfrozen' if args.unfreeze else 'frozen'})")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_mae = float('inf')
    best_model_state = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for key, value in val_metrics.items():
            if not np.isnan(value):
                print(f"  {key}: {value:.4f}")
        
        # Save best model
        current_mae = val_metrics.get('Overall_MAE', float('inf'))
        if current_mae < best_mae:
            best_mae = current_mae
            best_model_state = model.state_dict().copy()
            print(f"New best validation MAE: {best_mae:.4f}")
    
    # Load best model for final evaluation and submission
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (val MAE: {best_mae:.4f})")
    
    # Save model if requested
    if args.save_model:
        save_path = 'best_polymer_model.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    # Generate submission if test LMDB exists
    if os.path.exists(test_lmdb):
        print("\nGenerating submission...")
        generate_submission(model, test_lmdb, 'submission.csv', device)
    else:
        print(f"Test LMDB not found at {test_lmdb}")
        print("Skipping submission generation")

if __name__ == '__main__':
    main() 