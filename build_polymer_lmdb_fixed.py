#!/usr/bin/env python3
# build_polymer_lmdb_fixed.py
#   python build_polymer_lmdb_fixed.py train
#   python build_polymer_lmdb_fixed.py test

import os, sys, lmdb, tqdm, torch, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, HybridizationType
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from graphdata_lmdb import GraphData

# ───────────────────────────── args ───────────────────────────────
if len(sys.argv) != 2 or sys.argv[1] not in {"train", "test"}:
    print("usage:  python build_polymer_lmdb_fixed.py {train | test}")
    sys.exit(1)
SPLIT = sys.argv[1]  # "train" or "test"

# Detect environment
if os.path.exists('/kaggle'):
    DATA_ROOT = '/kaggle/input/neurips-open-polymer-prediction-2025'
    SAVE_DIR  = '/kaggle/working/processed_chunks'
else:
    DATA_ROOT = 'data'
    SAVE_DIR  = os.path.join(DATA_ROOT, "processed_chunks")

CSV_PATH  = os.path.join(DATA_ROOT, f"{SPLIT}.csv")
LMDB_OUT  = os.path.join(SAVE_DIR, f"polymer_{SPLIT}3d_dist.lmdb")

os.makedirs(SAVE_DIR, exist_ok=True)
if os.path.exists(LMDB_OUT):
    print(f"{LMDB_OUT} already exists – delete it to rebuild.")
    sys.exit(0)

# ─────────────────────────── constants ────────────────────────────
MAP_SIZE  = 2 * (1 << 30)   # 2 GiB
MAX_HOPS  = 3
MAX_NODES = 381             # generous PCQM-style padding

torch.manual_seed(0)
np.random.seed(0)

# ───────────────────────── helper functions ──────────────────────
def clean_polymer_smiles(smiles: str) -> str:
    """Replace polymer wildcard '*' with '[H]' to keep RDKit happy."""
    return smiles.replace('*', '[H]')

def rbf(d, K=32, D_min=0., D_max=10., beta=5.0):
    """Radial basis function expansion for edge distances."""
    d_np = d.numpy() if isinstance(d, torch.Tensor) else d
    d_np = np.clip(d_np, D_min, D_max)
    centers = np.linspace(D_min, D_max, K).reshape(-1, 1)
    vals    = d_np.reshape(1, -1)
    out = np.exp(-beta * (vals - centers)**2)
    return torch.FloatTensor(out.T)

def get_atom_features(atom):
    """Simple per-atom extras for geometry branch."""
    hyb = {
        HybridizationType.SP: 0,
        HybridizationType.SP2: 1,
        HybridizationType.SP3: 2,
        HybridizationType.SP3D: 3,
        HybridizationType.SP3D2: 4
    }
    return [
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        hyb.get(atom.GetHybridization(), -1),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
    ]

def safe_rdkit6(smiles: str):
    """Six RDKit molecular descriptors; robust to bad inputs."""
    try:
        m = Chem.MolFromSmiles(clean_polymer_smiles(smiles))
        if m is None:
            raise ValueError("MolFromSmiles returned None")
        return [
            Descriptors.MolWt(m),
            Descriptors.NumRotatableBonds(m),
            Descriptors.TPSA(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.NumHDonors(m),
            Descriptors.RingCount(m),
        ]
    except Exception as e:
        print(f"[safe_rdkit6] Warning on SMILES={smiles!r}: {e}")
        return [0.0]*6

def make_conformer(smiles: str, seed: int = 0, max_iters: int = 200):
    """ETKDG → UFF/MMFF; add/remove Hs appropriately for better coords."""
    try:
        base = clean_polymer_smiles(smiles)
        m = Chem.MolFromSmiles(base)
        if m is None:
            return None
        m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) == -1:
            return None
        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=max_iters)
        except RuntimeError:
            try:
                AllChem.MMFFOptimizeMolecule(m, mmffVariant='MMFF94', maxIters=max_iters)
            except Exception:
                pass
        m = Chem.RemoveHs(m)
        return m
    except Exception as e:
        print(f"[make_conformer] Warning for {smiles!r}: {e}")
        return None

@torch.no_grad()
def hop_distance(edge_index, num_nodes, max_dist=MAX_HOPS):
    """All-pairs min hop distance up to max_dist."""
    if num_nodes <= 1 or edge_index.size(1) == 0:
        return torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float().clamp_(0, 1)
    dist  = torch.full_like(A, float(max_dist))
    dist.fill_diagonal_(0.)
    reach = A.clone()
    for d in range(1, max_dist + 1):
        new = (reach > 0) & (dist == float(max_dist))
        dist.masked_fill_(new, float(d))
        reach = (reach @ A).clamp_(0, 1)
        if (dist < float(max_dist)).all():
            break
    return dist

# ───────────────────────── read CSV ───────────────────────────────
df = pd.read_csv(CSV_PATH)
assert 'id' in df.columns and 'SMILES' in df.columns, "CSV must have id and SMILES"
graph_ids = df['id'].astype(int).tolist()
smiles    = df['SMILES'].astype(str).tolist()
assert len(graph_ids) == len(smiles)

LABEL_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
store_lbl  = (SPLIT == "train")

print(f"→ will write {len(graph_ids):,} {SPLIT} graphs to {LMDB_OUT}")

# ───────────────────────── open LMDB ──────────────────────────────
env = lmdb.open(LMDB_OUT, map_size=MAP_SIZE)
txn = env.begin(write=True)

kept_ids   = []
n_ok       = 0
n_skipped  = 0
commit_every = 5_000

try:
    for i, (gid, smi) in enumerate(tqdm.tqdm(zip(graph_ids, smiles),
                                             total=len(smiles),
                                             desc=f"encode {SPLIT}")):
        try:
            # 1) 2-D graph (use cleaned SMILES for OGB)
            g = smiles2graph(clean_polymer_smiles(smi))
            data = Data(
                x          = torch.tensor(g['node_feat'],   dtype=torch.long),
                edge_index = torch.tensor(g['edge_index'],  dtype=torch.long),
                edge_attr  = torch.tensor(g['edge_feat'],   dtype=torch.float32),
            )

            # 2) 3-D geometry branch
            mol = make_conformer(smi)
            if mol is not None and mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                pos = torch.tensor([[conf.GetAtomPosition(a).x,
                                     conf.GetAtomPosition(a).y,
                                     conf.GetAtomPosition(a).z]
                                    for a in range(mol.GetNumAtoms())],
                                   dtype=torch.float32)
                dist_e = torch.norm(pos[data.edge_index[0]] - pos[data.edge_index[1]],
                                    dim=1, keepdim=True)
                data.pos = pos
                data.edge_attr = torch.cat([data.edge_attr, rbf(dist_e)], dim=1)  # +32
                data.extra_atom_feats = torch.tensor(
                    [get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32
                )
                data.has_xyz = torch.ones(1, dtype=torch.bool)
            else:
                E = data.edge_attr.size(0)
                data.pos = torch.zeros(data.x.size(0), 3, dtype=torch.float32)
                data.edge_attr = torch.cat([data.edge_attr, torch.zeros(E, 32)], dim=1)
                data.extra_atom_feats = torch.zeros(data.x.size(0), 5, dtype=torch.float32)
                data.has_xyz = torch.zeros(1, dtype=torch.bool)

            # 3) RDKit globals + labels
            data.rdkit_feats = torch.tensor(safe_rdkit6(smi), dtype=torch.float32).unsqueeze(0)  # (1,6)

            if store_lbl:
                vals = df.loc[i, LABEL_COLS].values
                vals = pd.to_numeric(vals, errors='coerce')  # np.nan where missing
                data.y = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)  # (1,5)
            else:
                data.y = torch.zeros(1, 5, dtype=torch.float32)

            # 4) hop-distance padding
            dist = hop_distance(data.edge_index, data.x.size(0))
            pad  = torch.full((MAX_NODES, MAX_NODES), MAX_HOPS, dtype=torch.float32)
            n    = dist.size(0)
            pad[:n, :n] = dist
            data.dist = pad.to(torch.uint8)

            # 5) write
            buf = GraphData.dump(GraphData.pack(data))
            txn.put(str(gid).encode(), buf)
            kept_ids.append(int(gid))
            n_ok += 1

            if (i + 1) % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)

        except Exception as e:
            n_skipped += 1
            print(f"[encode] Skipped id={gid} ({e}).")
            continue

    # Final commit and close
    txn.commit()
    env.sync()
    env.close()

    # Write the ids that actually exist in the LMDB
    idx_path = LMDB_OUT + ".ids.txt"
    with open(idx_path, "w") as f:
        for k in kept_ids:
            f.write(f"{k}\n")
    print(f"wrote {len(kept_ids)} ids to {idx_path}")
    print(f"Finished {SPLIT}: kept={n_ok}, skipped={n_skipped}")

    # Size
    try:
        sz = os.path.getsize(LMDB_OUT) / (1<<20)
        print(f"LMDB size: {sz:,.1f} MB")
    except OSError:
        pass

except Exception as e:
    print(f"Error creating LMDB: {e}")
    try:
        txn.abort()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass
    sys.exit(1)
