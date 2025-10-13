#!/usr/bin/env python3
# build_polymer_lmdb_debug.py
# Usage: python build_polymer_lmdb_debug.py <path_to_input_ids.txt>

import os, sys, lmdb, tqdm, torch, numpy as np, pandas as pd, re
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, HybridizationType
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from graphdata_lmdb import GraphData

# ───────────────────────────── args ───────────────────────────────
if len(sys.argv) != 2:
    print("usage: python build_polymer_lmdb_debug.py C:/Users/mattg/Downloads/kaggle_polymers_competition/data/processed_chunks/polymer_train3d_dist.lmdb.ids.txt")
    sys.exit(1)
INPUT_IDS_PATH = sys.argv[1]

# Detect environment and set paths
if os.path.exists('/kaggle'):
    DATA_ROOT = '/kaggle/input/neurips-open-polymer-prediction-2025'
    SAVE_DIR = '/kaggle/working/processed_chunks_debug'
    CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
else:
    DATA_ROOT = 'data'
    SAVE_DIR = os.path.join(DATA_ROOT, "processed_chunks_debug")
    CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

# ─────────────────────────── constants ────────────────────────────
MAP_SIZE = 2 * (1 << 30)
MAX_HOPS = 3
MAX_NODES = 381

torch.manual_seed(0)
np.random.seed(0)

# ───────────────────────── helper functions ──────────────────────
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
def rdkit_ogb_agree(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    from ogb.utils import smiles2graph
    return m.GetNumAtoms() == smiles2graph(smi)["num_nodes"]


def canonicalize_polymer_smiles(smiles: str, cap_atomic_num: int = 6) -> str:
    """
    Convert polymer SMILES with '*' attachment points into a normal, finite molecule
    by turning each '*' (dummy atom) into a carbon (default), preserving existing bonds.

    Returns a canonical isomeric SMILES with no explicit hydrogens.
    """
    # 1) Parse without sanitization so '*' and odd valences don't explode
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    rw = Chem.RWMol(mol)

    # 2) Turn every dummy ('*', atomic number 0) into a real atom (default: carbon)
    star_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in star_idxs:
        a = rw.GetAtomWithIdx(idx)
        a.SetAtomicNum(cap_atomic_num)    # 6 = carbon (robust for single/double bonds)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(False)            # allow RDKit to add implicit Hs as needed
        a.SetNumExplicitHs(0)

    mol2 = rw.GetMol()

    # 3) Sanitize; if aromatic kekulization is touchy, try a softer path
    try:
        Chem.SanitizeMol(mol2)
    except Exception:
        # Retry without kekulization then kekulize explicitly
        Chem.SanitizeMol(mol2, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.Kekulize(mol2, clearAromaticFlags=True)

    # 4) Remove explicit Hs so node counts match typical graph builders
    mol2 = Chem.RemoveHs(mol2)

    # 5) Canonical, isomeric SMILES keeps E/Z (/, \) when defined
    return Chem.MolToSmiles(mol2, isomericSmiles=True, canonical=True)


def rbf(d, K=32, D_min=0., D_max=10., beta=5.0):
    """Radial basis function expansion for edge distances."""
    d_np = d.numpy() if isinstance(d, torch.Tensor) else d
    d_np = np.clip(d_np, D_min, D_max)
    centers = np.linspace(D_min, D_max, K).reshape(-1, 1)
    vals = d_np.reshape(1, -1)
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
    """Six RDKit molecular descriptors; assume smiles is canonicalized & valid."""
    try:
        m = Chem.MolFromSmiles(smiles)
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
    """ETKDG → UFF/MMFF on a canonicalized SMILES."""
    try:
        m = Chem.MolFromSmiles(smiles)
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
    dist = torch.full_like(A, float(max_dist))
    dist.fill_diagonal_(0.)
    reach = A.clone()
    for d in range(1, max_dist + 1):
        new = (reach > 0) & (dist == float(max_dist))
        dist.masked_fill_(new, float(d))
        reach = (reach @ A).clamp_(0, 1)
        if (dist < float(max_dist)).all():
            break
    return dist

# ───────────────────────── read debug IDs ────────────────────────
try:
    with open(INPUT_IDS_PATH, 'r') as f:
        debug_ids = [int(line.strip()) for line in f if line.strip()]
except FileNotFoundError:
    print(f"Error: {INPUT_IDS_PATH} not found.")
    sys.exit(1)

# Read the full train CSV and filter for debug IDs
df = pd.read_csv(CSV_PATH)
df = df[df['id'].isin(debug_ids)]
if df.empty:
    print("No IDs from input list were found in the CSV. Exiting.")
    sys.exit(0)
df.set_index('id', inplace=True)

graph_ids = df.index.tolist()
smiles = df['SMILES'].tolist()

LMDB_OUT = os.path.join(SAVE_DIR, "polymer_debug.lmdb")
os.makedirs(SAVE_DIR, exist_ok=True)
if os.path.exists(LMDB_OUT):
    os.remove(LMDB_OUT) # Always rebuild debug LMDB
    print(f"Removed old {LMDB_OUT} to rebuild.")

LABEL_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
store_lbl = True # Always store labels for debugging

print(f"→ will write {len(graph_ids):,} debug graphs to {LMDB_OUT}")

# ───────────────────────── open LMDB ──────────────────────────────
env = lmdb.open(LMDB_OUT, map_size=MAP_SIZE)
txn = env.begin(write=True)

kept_ids = []
n_ok = 0
n_skipped = 0
commit_every = 1

try:
    for i, (gid, smi) in enumerate(tqdm.tqdm(zip(graph_ids, smiles),
                                             total=len(smiles),
                                             desc="encode debug")):
        try:
            print(f"\nProcessing ID: {gid}, SMILES: {smi}")
            
            # 1) Get molecule object to serve as a single source of truth for atom count
            canon_smi = canonicalize_polymer_smiles(smi)
            
            # (optional) assert agreement early
            if not rdkit_ogb_agree(canon_smi):
                raise ValueError("Atom/node mismatch even after canonicalization.")

            # RDKit 3D branch from the same canonical string
            mol_conf = make_conformer(canon_smi)
            if mol_conf is None:
                raise ValueError("RDKit Mol object could not be created or conformed.")

            # OGB graph from the same canonical string
            g = smiles2graph(canon_smi)

            
            # CRITICAL: Check for atom count consistency between RDKit and OGB.
            # Hard guard on counts
            num_nodes_ogb = g['num_nodes']
            num_atoms_mol = mol_conf.GetNumAtoms()
            if num_nodes_ogb != num_atoms_mol:
                raise ValueError(f"Atom count mismatch: OGB has {num_nodes_ogb} nodes, RDKit Mol has {num_atoms_mol}.")

            
            # Now that we know the counts are consistent, build the Data object
            data = Data(
                x=torch.tensor(g['node_feat'], dtype=torch.long),
                edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(g['edge_feat'], dtype=torch.float32),
            )

            # 2) 3-D geometry branch (now we know the mol object is valid)
            if mol_conf.GetNumConformers() > 0:
                print("Conformer generated successfully.")
                conf = mol_conf.GetConformer()
                pos = torch.tensor([[conf.GetAtomPosition(a).x,
                                     conf.GetAtomPosition(a).y,
                                     conf.GetAtomPosition(a).z]
                                    for a in range(mol_conf.GetNumAtoms())],
                                   dtype=torch.float32)
                
                # This line is now safe because we've checked for atom count consistency
                dist_e = torch.norm(pos[data.edge_index[0]] - pos[data.edge_index[1]], dim=1, keepdim=True)
                
                data.pos = pos
                data.edge_attr = torch.cat([data.edge_attr, rbf(dist_e)], dim=1)
                data.extra_atom_feats = torch.tensor(
                    [get_atom_features(a) for a in mol_conf.GetAtoms()], dtype=torch.float32
                )
                data.has_xyz = torch.ones(1, dtype=torch.bool)
            else:
                print("Warning: Conformer generation failed. Using zero-padded features.")
                E = data.edge_attr.size(0)
                data.pos = torch.zeros(data.x.size(0), 3, dtype=torch.float32)
                data.edge_attr = torch.cat([data.edge_attr, torch.zeros(E, 32)], dim=1)
                data.extra_atom_feats = torch.zeros(data.x.size(0), 5, dtype=torch.float32)
                data.has_xyz = torch.zeros(1, dtype=torch.bool)
                
            # 3) RDKit globals + labels
            data.rdkit_feats = torch.tensor(safe_rdkit6(smi), dtype=torch.float32).unsqueeze(0)
            vals = df.loc[gid, LABEL_COLS].values
            vals = pd.to_numeric(vals, errors='coerce')
            data.y = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

            # 4) hop-distance padding
            dist = hop_distance(data.edge_index, data.x.size(0))
            pad = torch.full((MAX_NODES, MAX_NODES), MAX_HOPS, dtype=torch.float32)
            n = dist.size(0)
            if n <= MAX_NODES:
                pad[:n, :n] = dist
            else:
                print(f"Warning: Graph has too many nodes ({n}). Padding with max hops.")
            data.dist = pad.to(torch.uint8)

            # 5) write
            buf = GraphData.dump(GraphData.pack(data))
            txn.put(str(gid).encode(), buf)
            kept_ids.append(int(gid))
            n_ok += 1
            txn.commit()
            txn = env.begin(write=True)

        except Exception as e:
            n_skipped += 1
            print(f"FAILED TO PROCESS ID={gid} ({e}).")
            txn.abort()
            env.close()
            sys.exit(1)

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
    print(f"Finished: kept={n_ok}, skipped={n_skipped}")

    # Size
    try:
        sz = os.path.getsize(LMDB_OUT) / (1 << 20)
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