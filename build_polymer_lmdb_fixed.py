#!/usr/bin/env python3
# build_polymer_lmdb_fixed.py
#   python build_polymer_lmdb_fixed.py train
#   python build_polymer_lmdb_fixed.py test

import os, sys, lmdb, tqdm, torch, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, HybridizationType, SanitizeFlags
from rdkit.Chem import rdMolDescriptors as rdmd
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from graphdata_lmdb import GraphData
from psmiles_canon import canonicalize as canonicalize_psmiles

# ------------------------------------- args -------------------------------------
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
    SAVE_DIR  = os.path.join(DATA_ROOT, "processed_chunks_new_CANON")

CSV_PATH  = os.path.join(DATA_ROOT, f"{SPLIT}.csv")
LMDB_OUT  = os.path.join(SAVE_DIR, f"polymer_{SPLIT}3d_dist.lmdb")

os.makedirs(SAVE_DIR, exist_ok=True)
if os.path.exists(LMDB_OUT):
    print(f"{LMDB_OUT} already exists - delete it to rebuild.")
    sys.exit(0)

# ------------------------------------- constants -------------------------------------
MAP_SIZE_INIT = 4 * (1 << 30) # 4 GiB
MAX_HOPS  = 3
MAX_NODES = 381 # generous PCQM-style padding

# Augmentation policy
N_CONFORMER_TRAIN = 10
N_CONFORMER_TEST  = 5
AUG_KEY_MULT = 1000 # key_id = parent_id * AUG_KEY_MULT + aug_idx
REPLACEMENT_Z = 6 # which atom to replace wildcard * with 

def n_aug_for_split(split: str) -> int:
    return N_CONFORMER_TRAIN if split == "train" else N_CONFORMER_TEST

torch.manual_seed(0)
np.random.seed(0)

# ------------------------------------- polymer normalization-------------------------------------
def _cap_then_rdkit_canon(s: str) -> str:
    return canonicalize_polymer_smiles(s, replace_atomic_num=REPLACEMENT_Z)[0]

def _psmiles_then_cap(s: str) -> str:
    try:
        s1 = canonicalize_psmiles(s) # unify and reduce
    except Exception:
        s1 = s
    try:
        return _cap_then_rdkit_canon(s1) # replace stars -> sanitize -> canonical SMILES
    except Exception:
        # last resort: RDKit canonicalize the unified string
        m = Chem.MolFromSmiles(s1)
        return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) if m else s1
    
def rdkit_ogb_agree(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    try:
        return m.GetNumAtoms() == smiles2graph(smi)["num_nodes"]
    except Exception:
        return False

def canonicalize_polymer_smiles(
    smiles: str,
    replace_atomic_num: int = 6, # 6=C, 7=N, 8=O, etc.
    record_star_idx: bool = True, # record positions of '*' before replacement
    kekulize_on_fallback: bool = True
):
    """
    Replace '*' (atomic #0) with a real atom for graph/3D stability, sanitize,
    and return canonical isomeric SMILES. Optionally record star indices.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    star_atom_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]

    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(replace_atomic_num)
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)

    mol2 = rw.GetMol()
    try:
        Chem.SanitizeMol(mol2)
    except Exception:
        # fallback without kekulization first, then try explicit kekulize if asked
        Chem.SanitizeMol(mol2, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
        if kekulize_on_fallback:
            Chem.Kekulize(mol2, clearAromaticFlags=True)

    # ensure defined stereochem after sanitize
    Chem.AssignStereochemistry(mol2, cleanIt=True, force=True)

    # explicit Hydrogens confuse canonical SMILES comparisons, so I remove for stability
    mol2 = Chem.RemoveHs(mol2)

    can = Chem.MolToSmiles(mol2, isomericSmiles=True, canonical=True)
    meta = {"star_indices": star_atom_ids} if record_star_idx else {}
    return can, meta

# ------------------------------------- helper functions -------------------------------------
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

def safe_rdkit15(smiles_str: str):
    """15 stable RDKit descriptors; input should already be canonicalized SMILES."""
    try:
        # if a tuple sneaks in, take the string
        if isinstance(smiles_str, tuple):
            smiles_str = smiles_str[0]

        m = Chem.MolFromSmiles(smiles_str)
        if m is None:
            raise ValueError("MolFromSmiles returned None")

        heavy = Descriptors.HeavyAtomCount(m)
        arom_atoms = sum(a.GetIsAromatic() for a in m.GetAtoms())
        hetero = sum(a.GetAtomicNum() not in (1, 6) for a in m.GetAtoms())
        aromatic_prop = (arom_atoms / max(1, heavy))
        hetero_frac = (hetero / max(1, heavy))

        return [
            Descriptors.MolWt(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.NumHDonors(m),
            Descriptors.RingCount(m),
            Descriptors.MolMR(m),
            Descriptors.MolLogP(m),                
            rdmd.CalcFractionCSP3(m),
            rdmd.CalcNumAromaticRings(m),
            aromatic_prop,
            Descriptors.TPSA(m),
            Descriptors.NumRotatableBonds(m),
            hetero_frac,
            rdmd.CalcLabuteASA(m),
            rdmd.CalcKappa1(m),
            rdmd.CalcKappa2(m),
        ]
    except Exception as e:
        print(f"[safe_rdkit15] Warning on SMILES={smiles_str!r}: {e}")
        return [0.0] * 15


def seed_for(parent_id: int, aug_idx: int) -> int:
    # 32-bit deterministic seed from ids
    return (parent_id * 1315423911 + aug_idx * 2654435761) & 0x7fffffff

def make_one_conformer(smiles: str, seed: int, max_iters: int = 300):
    """
    Single ETKDG try (with/without random coords) using a fixed seed.
    If embedding fails, return 2D Mol (no conformer) as fallback.
    """
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        return None
    for use_random in (True, False):
        m = Chem.AddHs(Chem.Mol(base))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = use_random
        params.enforceChirality = True
        params.pruneRmsThresh = -1.0
        res = AllChem.EmbedMolecule(m, params)
        if res != -1:
            try:
                AllChem.UFFOptimizeMolecule(m, maxIters=max_iters)
            except Exception:
                try:
                    AllChem.MMFFOptimizeMolecule(m, mmffVariant='MMFF94', maxIters=max_iters)
                except Exception:
                    pass
            return Chem.RemoveHs(m)
    return Chem.RemoveHs(base)

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

# ------------------------------------- read CSV -------------------------------------
df = pd.read_csv(CSV_PATH)
assert 'id' in df.columns and 'SMILES' in df.columns, "CSV must have id and SMILES"

# Precompute canonical SMILES once
# Start with raw PSMILES strings in df['SMILES'] (two stars form)
df['SMILES_pscanon'] = df['SMILES'].astype(str).map(canonicalize_psmiles)

# Final canonical string used downstream / stored in LMDB
df['SMILES_canon'] = df['SMILES_pscanon'].map(_psmiles_then_cap)

# Index by id so labels align by id (not row position)
df.set_index('id', inplace=True)

graph_ids = df.index.astype(int).tolist()
smiles = df['SMILES_canon'].tolist()
assert len(graph_ids) == len(smiles)

LABEL_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
store_lbl = (SPLIT == "train")
N_AUG = n_aug_for_split(SPLIT)

print(f"-> will write {len(graph_ids):,} x {N_AUG} {SPLIT} graphs to {LMDB_OUT}")
print(f"Wildcard replacement atom Z = {REPLACEMENT_Z}")

# ------------------------------------- open LMDB -------------------------------------
env = lmdb.open(LMDB_OUT, map_size=MAP_SIZE_INIT, subdir=True, lock=True)
current_map_size = MAP_SIZE_INIT
txn = env.begin(write=True)

def put_with_resize(txn, key_bytes, value_bytes):
    """Put with automatic map growth on MapFullError."""
    global current_map_size
    try:
        txn.put(key_bytes, value_bytes)
        return txn
    except lmdb.MapFullError:
        # Grow map and retry
        txn.abort()
        new_size = int(current_map_size * 2)
        env.set_mapsize(new_size)
        current_map_size = new_size
        print(f"[LMDB] Map full -> resizing to {current_map_size/(1<<30):.1f} GiB")
        txn = env.begin(write=True)
        txn.put(key_bytes, value_bytes)
        return txn

kept_ids = []
parent_rows = []  # for parent_map.tsv
parent_meta_rows = []     
n_ok = 0
n_skipped = 0
commit_every = 5_000

try:
    pbar = tqdm.tqdm(zip(graph_ids, smiles), total=len(smiles), desc=f"encode {SPLIT}")
    for i, (gid, smi) in enumerate(pbar):
        try:
            # early agreement guard
            if not rdkit_ogb_agree(smi):
                raise ValueError("Atom/node mismatch even after canonicalization.")

            # 1) 2-D graph (canonicalized SMILES for OGB) – same for all augs
            g = smiles2graph(smi)
            base_data = Data(
                x = torch.tensor(g['node_feat'],   dtype=torch.long),
                edge_index = torch.tensor(g['edge_index'],  dtype=torch.long),
                edge_attr = torch.tensor(g['edge_feat'],   dtype=torch.float32),
            )
            assert int(g['num_nodes']) == base_data.x.size(0)
            # Compute size/meta once per parent
            raw_smi = df.loc[gid, 'SMILES'] # original, not canonicalized
            n_atoms_2d = int(g['num_nodes'])
            ps = df.loc[gid, 'SMILES_pscanon'] if 'SMILES_pscanon' in df.columns else df.loc[gid, 'SMILES']
            star_count = ps.count('[*]') # after the normalization this is correct
            replacement_Z = REPLACEMENT_Z # current default

            # keep on base_data so it travels into every augmented child
            base_data.n_atoms_2d = torch.tensor([n_atoms_2d], dtype=torch.int32)
            base_data.star_count = torch.tensor([star_count], dtype=torch.int16)
            base_data.replacement_Z = torch.tensor([replacement_Z], dtype=torch.int16)
            # add a CSV row (once per parent)
            parent_meta_rows.append((int(gid), n_atoms_2d, star_count, replacement_Z))
            # hop-distance (same for all augs)
            dist = hop_distance(base_data.edge_index, base_data.x.size(0))
            pad = torch.full((MAX_NODES, MAX_NODES), MAX_HOPS, dtype=torch.float32)
            n = dist.size(0)
            if n <= MAX_NODES:
                pad[:n, :n] = dist
            base_data.dist = pad.to(torch.uint8)

            # RDKit globals + labels (same for all augs)
            base_data.rdkit_feats = torch.tensor(safe_rdkit15(smi), dtype=torch.float32).unsqueeze(0)
            if store_lbl:
                vals = df.loc[gid, LABEL_COLS].values
                vals = pd.to_numeric(vals, errors='coerce')
                base_data.y = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
            else:
                base_data.y = torch.zeros(1, 5, dtype=torch.float32)

            # write K augmented entries
            for aug_idx in range(N_AUG):
                seed = seed_for(int(gid), aug_idx)
                mol = make_one_conformer(smi, seed)

                if mol is None or g['num_nodes'] != mol.GetNumAtoms():
                    # keep but zero pad geometry
                    data = base_data.clone()
                    E = data.edge_attr.size(0)
                    data.pos = torch.zeros(data.x.size(0), 3, dtype=torch.float32)
                    data.edge_attr = torch.cat([data.edge_attr, torch.zeros(E, 32)], dim=1)
                    data.extra_atom_feats = torch.zeros(data.x.size(0), 5, dtype=torch.float32)
                    data.has_xyz = torch.zeros(1, dtype=torch.bool)
                else:
                    data = base_data.clone()
                    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
                    if conf is not None:
                        pos = torch.tensor([[conf.GetAtomPosition(a).x,
                                             conf.GetAtomPosition(a).y,
                                             conf.GetAtomPosition(a).z]
                                            for a in range(mol.GetNumAtoms())], dtype=torch.float32)
                    else:
                        pos = torch.zeros(data.x.size(0), 3, dtype=torch.float32)
                    data.pos = pos
                    dist_e = torch.norm(pos[data.edge_index[0]] - pos[data.edge_index[1]], dim=1, keepdim=True)
                    data.edge_attr = torch.cat([data.edge_attr, rbf(dist_e)], dim=1)  # +32
                    data.extra_atom_feats = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()],
                                                         dtype=torch.float32)
                    data.has_xyz = torch.ones(1, dtype=torch.bool)

                # meta for traceability
                data.parent_id = torch.tensor([int(gid)], dtype=torch.int64)
                data.aug_idx = torch.tensor([aug_idx], dtype=torch.int64)
                data.seed = torch.tensor([seed], dtype=torch.int64)

                # synthetic key
                key_id = int(gid) * AUG_KEY_MULT + aug_idx
                buf = GraphData.dump(GraphData.pack(data))
                key_bytes = str(key_id).encode()

                txn = put_with_resize(txn, key_bytes, buf)
                kept_ids.append(key_id)
                parent_rows.append((key_id, int(gid), aug_idx, seed))
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
    # Parent map for ensembling / averaging
    pmap_path = LMDB_OUT + ".parent_map.tsv"
    with open(pmap_path, "w") as f:
        f.write("key_id\tparent_id\taug_idx\tseed\n")
        for row in parent_rows:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")

    print(f"wrote {len(kept_ids)} ids to {idx_path}")
    print(f"wrote parent map to {pmap_path}")
    print(f"Finished {SPLIT}: kept={n_ok}, skipped={n_skipped}")
    meta_parent_path = LMDB_OUT + ".parent_meta.tsv"
    with open(meta_parent_path, "w") as f:
        f.write("parent_id\tn_atoms_2d\tstar_count\treplacement_Z\n")
        seen = set()
        for row in parent_meta_rows:
            pid = row[0]
            if pid in seen: 
                continue
            seen.add(pid)
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")
    print(f"wrote parent meta to {meta_parent_path}")

    # Size
    try:
        sz = os.path.getsize(LMDB_OUT) / (1<<20)
        print(f"LMDB size: {sz:,.1f} MB (map limit {current_map_size/(1<<30):.1f} GiB)")
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