#!/usr/bin/env python3
# Merge supplement CSVs into train.csv using RAW string SMILES matching.
# No canonicalization. Base values win; only fill NaNs. New SMILES get new ids.

import os, argparse
import pandas as pd
import numpy as np

LABEL_COLS = ['Tg','FFV','Tc','Density','Rg']

def load_base(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {'id','SMILES'}
    assert req.issubset(df.columns), f"Base train.csv must have columns {req}"
    # Ensure full label schema
    for c in LABEL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # Types and minimal cleanup (strip only; NO canonicalization)
    df['id'] = df['id'].astype(int)
    df['SMILES'] = df['SMILES'].astype(str).str.strip()
    for c in LABEL_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Drop fully empty SMILES rows, if any
    df = df.dropna(subset=['SMILES'])
    return df[['id','SMILES'] + LABEL_COLS].copy()

def load_supp(path: str, target: str, value_names_hint=()) -> pd.DataFrame:
    """Load supplement: first col = SMILES, one numeric value column -> target."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['SMILES', target])

    df = pd.read_csv(path)
    assert df.shape[1] >= 2, f"{path} must have at least two columns (SMILES, value)"
    smiles_col = df.columns[0]
    df = df.rename(columns={smiles_col: 'SMILES'})
    df['SMILES'] = df['SMILES'].astype(str).str.strip()

    # Pick value col:
    # 1) exact matches from hint
    for name in value_names_hint:
        if name in df.columns:
            val_col = name; break
    else:
        # 2) any numeric column after SMILES
        numeric_cols = [c for c in df.columns if c != 'SMILES' and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            # 3) fallback: coerce all non-SMILES and take the first numeric-looking
            for c in df.columns:
                if c == 'SMILES': continue
                df[c] = pd.to_numeric(df[c], errors='coerce')
            numeric_cols = [c for c in df.columns if c != 'SMILES' and pd.api.types.is_numeric_dtype(df[c])]
            assert len(numeric_cols) > 0, f"{path}: no numeric value column found for {target}"
        val_col = numeric_cols[0]

    sub = df[['SMILES', val_col]].rename(columns={val_col: target}).copy()
    sub[target] = pd.to_numeric(sub[target], errors='coerce')
    sub = sub.dropna(subset=['SMILES', target])
    # If duplicates in supplement: average by SMILES (raw string key)
    sub = sub.groupby('SMILES', as_index=False).agg({target: 'mean'})
    return sub

def merge_all(base_path: str, d1_path: str, d3_path: str, d4_path: str, out_path: str):
    base = load_base(base_path)

    # Build SMILES -> row index map (keep first occurrence if duplicates)
    smiles_to_idx = {}
    for idx, s in base['SMILES'].items():
        if s not in smiles_to_idx:
            smiles_to_idx[s] = idx

    # Load supplements (dataset1 -> Tc from TC_mean; dataset3 -> Tg; dataset4 -> FFV)
    d1 = load_supp(d1_path, 'Tc',  value_names_hint=('TC_mean','Tc','tc_mean','tc'))
    d3 = load_supp(d3_path, 'Tg',  value_names_hint=('Tg','tg'))
    d4 = load_supp(d4_path, 'FFV', value_names_hint=('FFV','ffv'))

    filled = {'Tc': 0, 'Tg': 0, 'FFV': 0}
    pending_new = {}  # SMILES -> dict with any of Tg/FFV/Tc set

    def fill_or_queue(smiles: str, target: str, value: float):
        if smiles in smiles_to_idx:
            i = smiles_to_idx[smiles]
            if pd.isna(base.at[i, target]):
                base.at[i, target] = float(value)
                filled[target] += 1
        else:
            rec = pending_new.setdefault(smiles, {'SMILES': smiles, 'Tg': np.nan, 'FFV': np.nan, 'Tc': np.nan, 'Density': np.nan, 'Rg': np.nan})
            rec[target] = float(value)

    for _, r in d1.iterrows():
        fill_or_queue(r['SMILES'], 'Tc',  r['Tc'])
    for _, r in d3.iterrows():
        fill_or_queue(r['SMILES'], 'Tg',  r['Tg'])
    for _, r in d4.iterrows():
        fill_or_queue(r['SMILES'], 'FFV', r['FFV'])

    # Append new SMILES rows with fresh, sequential ids
    appended = 0
    if pending_new:
        next_id = int(base['id'].max()) + 1 if len(base) else 0
        adds = []
        for s, payload in pending_new.items():
            row = {'id': next_id, 'SMILES': payload['SMILES'],
                   'Tg': payload['Tg'], 'FFV': payload['FFV'], 'Tc': payload['Tc'],
                   'Density': payload['Density'], 'Rg': payload['Rg']}
            adds.append(row); next_id += 1
        base = pd.concat([base, pd.DataFrame(adds)], ignore_index=True)
        appended = len(adds)

    # Finalize and write
    base = base[['id','SMILES'] + LABEL_COLS].copy()
    base['id'] = base['id'].astype(int)
    for c in LABEL_COLS: base[c] = pd.to_numeric(base[c], errors='coerce')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base.to_csv(out_path, index=False)

    print("=== Merge summary (raw match) ===")
    print(f"Filled -> Tg: {filled['Tg']}, FFV: {filled['FFV']}, Tc: {filled['Tc']}")
    print(f"Appended new molecules: {appended}")
    print(f"Wrote: {out_path}  (rows: {len(base):,})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Path to base train.csv')
    ap.add_argument('--d1', default='', help='Path to dataset1.csv (SMILES + TC_mean -> Tc)')
    ap.add_argument('--d3', default='', help='Path to dataset3.csv (SMILES + Tg -> Tg)')
    ap.add_argument('--d4', default='', help='Path to dataset4.csv (SMILES + FFV -> FFV)')
    ap.add_argument('--out', required=True, help='Output path for train_supplemented.csv')
    args = ap.parse_args()
    merge_all(args.base, args.d1, args.d3, args.d4, args.out)

if __name__ == "__main__":
    main()
