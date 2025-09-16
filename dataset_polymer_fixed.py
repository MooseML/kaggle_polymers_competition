# dataset_polymer_fixed.py ----------------------------------------------------------
import os
import torch
import pandas as pd
import numpy as np
import lmdb
import pickle
import lz4.frame as lz4f
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, HybridizationType
from torch_geometric.data import Data, Dataset
from ogb.utils import smiles2graph
from torch_geometric.utils import to_dense_adj

# Constants
MAX_HOPS = 3
MAX_NODES = 381  # PCQM padding – plenty for these polymers

def safe_rdkit6(smiles):
    """Compute 6 RDKit molecular descriptors with error handling"""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError("Failed to parse SMILES")
        return [Descriptors.MolWt(m), Descriptors.NumRotatableBonds(m),
                Descriptors.TPSA(m),   Descriptors.NumHAcceptors(m),
                Descriptors.NumHDonors(m), Descriptors.RingCount(m)]
    except Exception as e:
        print(f"Warning: Failed to compute RDKit descriptors for SMILES: {smiles}")
        print(f"Error: {str(e)}")
        return [0.0] * 6  # Return zeros as fallback

def make_conformer(smiles: str, seed: int = 0, max_iters: int = 200):
    """Generate 3D conformer using ETKDG"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    
    # ETKDG embedding with correct params
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(m, params) == -1:
        return None
    
    # Optimize with UFF/MMFF
    try:
        AllChem.UFFOptimizeMolecule(m, maxIters=max_iters)
    except RuntimeError:
        try:
            AllChem.MMFFOptimizeMolecule(m, mmffVariant='MMFF94', maxIters=max_iters)
        except Exception:
            pass  # keep raw ETKDG coords
    
    return m

def rbf(d, K=32, D_min=0., D_max=10., beta=5.0):
    """Radial basis function expansion"""
    d_np = d.numpy() if isinstance(d, torch.Tensor) else d
    d_np = np.clip(d_np, D_min, D_max)
    d_mu = np.linspace(D_min, D_max, K)
    d_mu = d_mu.reshape(-1, 1)
    d_np = d_np.reshape(1, -1)
    rbf = np.exp(-beta * (d_np - d_mu)**2)
    return torch.FloatTensor(rbf.T)

def get_atom_features(atom):
    """Extract atom features"""
    hybridization_types = {
        HybridizationType.SP: 0,
        HybridizationType.SP2: 1,
        HybridizationType.SP3: 2,
        HybridizationType.SP3D: 3,
        HybridizationType.SP3D2: 4
    }
    return [
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        hybridization_types.get(atom.GetHybridization(), -1),
        int(atom.IsInRing()),
        atom.GetTotalNumHs()
    ]

@torch.no_grad()
def hop_distance(edge_index, num_nodes, max_dist=MAX_HOPS):
    """Calculate minimum hop distances between all nodes"""
    if num_nodes <= 1 or edge_index.size(1) == 0:
        return torch.zeros(num_nodes, num_nodes, device=edge_index.device)

    # 0/1 adjacency in float32
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()
    A = A.clamp_(0, 1)  # be safe
    dist = torch.full_like(A, max_dist)
    dist.fill_diagonal_(0.)
    reach = A.clone()  # paths of length 1
    
    for d in range(1, max_dist + 1):
        # nodes discovered for the first time at hop‑d
        new = (reach > 0) & (dist == max_dist)
        dist.masked_fill_(new, float(d))
        
        # front_{d+1} = front_d @ A (float matmul)
        reach = (reach @ A).clamp_(0, 1)
    
    return dist


# Standalone GraphData class to avoid imports
class GraphData(Data):
    def __cat_dim__(self, key, value, *a, **k):
        return None if key=="dist" else super().__cat_dim__(key,value,*a,**k)

    @staticmethod
    def dump(obj):
        return lz4f.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL),
                            compression_level=0,     # fastest
                            block_linked=True)
    @staticmethod
    def undump(buf: bytes):
        return pickle.loads(lz4f.decompress(buf))

    @staticmethod
    def pack(d):
        return dict(
            x=d.x.numpy().astype(np.int16),
            edge_index=d.edge_index.numpy().astype(np.int32),
            edge_attr=d.edge_attr.numpy().astype(np.float32),
            pos=d.pos.numpy().astype(np.float32),
            extra_atom=d.extra_atom_feats.numpy().astype(np.float32),
            rdkit=d.rdkit_feats.numpy().astype(np.float32),
            y=d.y.numpy().astype(np.float32),
            has_xyz=bool(d.has_xyz.item()),
            dist=d.dist.numpy().astype(np.uint8)
        )

    @staticmethod
    def dict2data(d):
        return GraphData(           # ← use the subclass!
            x               = torch.from_numpy(d['x']).long(),
            edge_index      = torch.from_numpy(d['edge_index']).long(),
            edge_attr       = torch.from_numpy(d['edge_attr']).float(),
            pos             = torch.from_numpy(d['pos']),
            extra_atom_feats= torch.from_numpy(d['extra_atom']),
            rdkit_feats     = torch.from_numpy(d['rdkit']),
            y               = torch.from_numpy(d['y']),
            has_xyz         = torch.tensor([d['has_xyz']], dtype=torch.bool),
            dist            = torch.from_numpy(d['dist']).float(),  # uint8 → fp32 (N, N)
        )

from torch.utils.data import Dataset
from graphdata_lmdb import GraphData

class LMDBDataset(Dataset):
    def __init__(self, ids, path, strict: bool = True):
        """
        ids: iterable of int-like ids (from CSV)
        path: path to the LMDB directory
        strict: if True, raise on missing keys; if False, drop missing keys on the fly
        """
        self.ids  = [int(i) for i in ids]
        self.path = path
        self.env  = None
        self.strict = strict

        # Prefer companion ids file (written by the builder); fallback to scanning LMDB once.
        idx_path = self.path + ".ids.txt"
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                good = set(int(line.strip()) for line in f if line.strip())
            source = "ids.txt"
        else:
            env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
            with env.begin(buffers=True) as txn:
                cur = txn.cursor()
                good = set(int(k.decode()) for k, _ in cur)
            env.close()
            source = "scan"

        before = len(self.ids)
        self.ids = [i for i in self.ids if i in good]
        dropped = before - len(self.ids)
        if dropped:
            print(f"[LMDBDataset] Dropped {dropped} ids not found in LMDB ({source}).")

    # Prevent LMDB env from being pickled to worker processes
    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.env is None:
            # Open per-process, lazily (works on Windows spawn or Unix fork)
            self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)

        gid = self.ids[index]
        with self.env.begin(buffers=True) as txn:
            buf = txn.get(str(gid).encode())

        if buf is None:
            if self.strict:
                raise KeyError(f"Key {gid} not found in LMDB at {self.path}")
            # Non-strict mode: drop missing id and try another
            del self.ids[index]
            if not self.ids:
                raise KeyError("All ids were dropped; LMDB is empty or mismatched.")
            return self[index % len(self.ids)]

        return GraphData.dict2data(GraphData.undump(bytes(buf)))



class PolymerCSV(Dataset):
    """
    On-the-fly dataset that converts SMILES → 3D graph at runtime.
    Slower than LMDB but useful for debugging.
    """
    def __init__(self, csv_path, split='train', transform=None, seed=0):
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        self.seed = seed
        
        self.df = pd.read_csv(csv_path)
        self.label_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.store_labels = split == 'train'
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['SMILES']
        graph_id = row['id']
        
        # 1) Basic 2D graph
        g = smiles2graph(smiles)
        data = Data(
            x = torch.tensor(g['node_feat'], dtype=torch.long),
            edge_index = torch.tensor(g['edge_index'], dtype=torch.long),
            edge_attr = torch.tensor(g['edge_feat'], dtype=torch.float32)
        )
        
        # 2) 3D geometry (ETKDG)
        mol = make_conformer(smiles, seed=self.seed + idx)
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            pos = torch.tensor([[conf.GetAtomPosition(a).x,
                               conf.GetAtomPosition(a).y,
                               conf.GetAtomPosition(a).z]
                              for a in range(mol.GetNumAtoms())],
                             dtype=torch.float32)
            dist = torch.norm(pos[data.edge_index[0]] -
                            pos[data.edge_index[1]], dim=1, keepdim=True)
            data.pos = pos
            data.edge_attr = torch.cat([data.edge_attr, rbf(dist)], 1)  # +32
            data.extra_atom_feats = torch.tensor(
                [get_atom_features(a) for a in mol.GetAtoms()],
                dtype=torch.float32)
            data.has_xyz = torch.ones(1, dtype=torch.bool)
        else:
            # Fallback - no 3D
            E = data.edge_attr.size(0)
            data.pos = torch.zeros(data.x.size(0), 3)
            data.edge_attr = torch.cat([data.edge_attr, torch.zeros(E, 32)], 1)
            data.extra_atom_feats = torch.zeros(data.x.size(0), 5)
            data.has_xyz = torch.zeros(1, dtype=torch.bool)
        
        # 3) RDKit globals + labels
        data.rdkit_feats = torch.tensor(safe_rdkit6(smiles), dtype=torch.float32)
        if self.store_labels:
            data.y = torch.tensor(row[self.label_cols].values, dtype=torch.float32)
        else:
            data.y = torch.zeros(5)
        
        # 4) Hop distance padding
        dist = hop_distance(data.edge_index, data.x.size(0))
        pad = torch.full((MAX_NODES, MAX_NODES), MAX_HOPS, dtype=dist.dtype)
        n = dist.size(0)
        pad[:n, :n] = dist
        data.dist = pad.to(torch.uint8)
        
        if self.transform:
            data = self.transform(data)
            
        return data 