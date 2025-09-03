# # build_polymer_lmdb.py  ────────────────────────────────────────────
# #   python build_polymer_lmdb.py train
# #   python build_polymer_lmdb.py test
# #
# # Writes →  polymer_train3d_dist.lmdb  (~40 MB)
# #           polymer_test3d_dist.lmdb   (~35 MB)
# # -------------------------------------------------------------------

# import os, sys, lmdb, tqdm, torch, numpy as np, pandas as pd
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, HybridizationType
# from ogb.utils import smiles2graph
# from torch_geometric.data import Data
# from torch_geometric.utils import to_dense_adj
# from graphdata_lmdb import GraphData

# # ───────────────────────────── args ───────────────────────────────
# if len(sys.argv) != 2 or sys.argv[1] not in {"train", "test"}:
#     print("usage:  python build_polymer_lmdb.py {train | test}")
#     sys.exit(1)
# SPLIT = sys.argv[1]                                   # csv name (= split)

# # Detect Kaggle environment
# if os.path.exists('/kaggle'):
#     DATA_ROOT = '/kaggle/input/neurips-open-polymer-prediction-2025'
#     SAVE_DIR  = '/kaggle/working/processed_chunks'
# else:
#     DATA_ROOT = 'data'
#     SAVE_DIR  = os.path.join(DATA_ROOT, "processed_chunks")

# CSV_PATH  = os.path.join(DATA_ROOT, f"{SPLIT}.csv")
# LMDB_OUT  = os.path.join(SAVE_DIR, f"polymer_{SPLIT}3d_dist.lmdb")

# os.makedirs(SAVE_DIR, exist_ok=True)
# if os.path.exists(LMDB_OUT):
#     print(f"{LMDB_OUT} already exists – delete it to rebuild.")
#     sys.exit(0)

# # ─────────────────────────── constants ────────────────────────────
# MAP_SIZE  = 2 * (1 << 30)   # 2 GiB (over-kill, but safe)
# MAX_HOPS  = 3
# MAX_NODES = 381  # PCQM padding – plenty for these polymers

# torch.manual_seed(0);  np.random.seed(0)

# # ───────────────────────── helper functions ──────────────────────
# def rbf(d, K=32, D_min=0., D_max=10., beta=5.0):
#     """Radial basis function expansion"""
#     d_np = d.numpy() if isinstance(d, torch.Tensor) else d
#     d_np = np.clip(d_np, D_min, D_max)
#     d_mu = np.linspace(D_min, D_max, K)
#     d_mu = d_mu.reshape(-1, 1)
#     d_np = d_np.reshape(1, -1)
#     rbf = np.exp(-beta * (d_np - d_mu)**2)
#     return torch.FloatTensor(rbf.T)

# def make_conformer(smiles: str, seed: int = 0, max_iters: int = 200):
#     """Generate 3D conformer using ETKDG"""
#     m = Chem.MolFromSmiles(smiles)
#     if m is None:
#         return None
    
#     # ETKDG embedding with correct params
#     params = AllChem.ETKDGv3()
#     params.randomSeed = seed
#     params.useRandomCoords = True
#     if AllChem.EmbedMolecule(m, params) == -1:
#         return None
    
#     # Optimize with UFF/MMFF
#     try:
#         AllChem.UFFOptimizeMolecule(m, maxIters=max_iters)
#     except RuntimeError:
#         try:
#             AllChem.MMFFOptimizeMolecule(m, mmffVariant='MMFF94', maxIters=max_iters)
#         except Exception:
#             pass  # keep raw ETKDG coords
    
#     return m

# def get_atom_features(atom):
#     """Extract atom features"""
#     hybridization_types = {
#         HybridizationType.SP: 0,
#         HybridizationType.SP2: 1,
#         HybridizationType.SP3: 2,
#         HybridizationType.SP3D: 3,
#         HybridizationType.SP3D2: 4
#     }
#     return [
#         atom.GetFormalCharge(),
#         int(atom.GetIsAromatic()),
#         hybridization_types.get(atom.GetHybridization(), -1),
#         int(atom.IsInRing()),
#         atom.GetTotalNumHs()
#     ]

# def safe_rdkit6(smiles):
#     """Compute 6 RDKit molecular descriptors with error handling"""
#     try:
#         m = Chem.MolFromSmiles(smiles)
#         if m is None:
#             raise ValueError("Failed to parse SMILES")
#         return [Descriptors.MolWt(m), Descriptors.NumRotatableBonds(m),
#                 Descriptors.TPSA(m),   Descriptors.NumHAcceptors(m),
#                 Descriptors.NumHDonors(m), Descriptors.RingCount(m)]
#     except Exception as e:
#         print(f"Warning: Failed to compute RDKit descriptors for SMILES: {smiles}")
#         print(f"Error: {str(e)}")
#         return [0.0] * 6  # Return zeros as fallback

# @torch.no_grad()
# def hop_distance(edge_index, num_nodes, max_dist=MAX_HOPS):
#     """Calculate minimum hop distances between all nodes"""
#     if num_nodes <= 1 or edge_index.size(1) == 0:
#         return torch.zeros(num_nodes, num_nodes, device=edge_index.device)

#     # 0/1 adjacency in float32
#     A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()
#     A = A.clamp_(0, 1)  # be safe
#     dist = torch.full_like(A, max_dist)
#     dist.fill_diagonal_(0.)
#     reach = A.clone()  # paths of length 1
    
#     for d in range(1, max_dist + 1):
#         # nodes discovered for the first time at hop‑d
#         new = (reach > 0) & (dist == max_dist)
#         dist.masked_fill_(new, float(d))
        
#         # front_{d+1} = front_d @ A (float matmul)
#         reach = (reach @ A).clamp_(0, 1)
        
#         # early exit: all reachable
#         if (dist < max_dist).all():
#             break
    
#     return dist

# # ───────────────────────── read CSV ───────────────────────────────
# df = pd.read_csv(CSV_PATH)
# smiles = df['SMILES'].tolist()
# graph_ids = df['id'].astype(int).tolist()

# LABEL_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
# store_lbl = SPLIT == "train"          # test split has no labels

# print(f"→ will write {len(graph_ids):,} {SPLIT} graphs to {LMDB_OUT}")

# # ───────────────────────── open LMDB ──────────────────────────────
# env = lmdb.open(LMDB_OUT, map_size=MAP_SIZE)
# txn = env.begin(write=True)

# try:
#     for i, (gid, smi) in enumerate(tqdm.tqdm(zip(graph_ids, smiles),
#                                              total=len(smiles),
#                                              desc=f"encode {SPLIT}")):
#         # 1) basic 2-D graph ------------------------------------------------
#         g = smiles2graph(smi)
#         data = Data(
#             x = torch.tensor(g['node_feat'], dtype=torch.long),
#             edge_index = torch.tensor(g['edge_index'], dtype=torch.long),
#             edge_attr = torch.tensor(g['edge_feat'], dtype=torch.float32)
#         )

#         # 2) geometry branch (ETKDG) ---------------------------------------
#         mol = make_conformer(smi)
#         if mol is not None and mol.GetNumConformers() > 0:
#             conf = mol.GetConformer()
#             pos = torch.tensor([[conf.GetAtomPosition(a).x,
#                                conf.GetAtomPosition(a).y,
#                                conf.GetAtomPosition(a).z]
#                               for a in range(mol.GetNumAtoms())],
#                              dtype=torch.float32)
#             dist = torch.norm(pos[data.edge_index[0]] -
#                             pos[data.edge_index[1]], dim=1, keepdim=True)
#             data.pos = pos
#             data.edge_attr = torch.cat([data.edge_attr, rbf(dist)], 1)  # +32
#             data.extra_atom_feats = torch.tensor(
#                 [get_atom_features(a) for a in mol.GetAtoms()],
#                 dtype=torch.float32)
#             data.has_xyz = torch.ones(1, dtype=torch.bool)
#         else:                              # geometry-free fallback
#             E = data.edge_attr.size(0)
#             data.pos = torch.zeros(data.x.size(0), 3)
#             data.edge_attr = torch.cat([data.edge_attr,
#                                       torch.zeros(E, 32)], 1)
#             data.extra_atom_feats = torch.zeros(data.x.size(0), 5)
#             data.has_xyz = torch.zeros(1, dtype=torch.bool)

#         # 3) RDKit globals + LABEL -----------------------------------------
#         data.rdkit_feats = torch.tensor(safe_rdkit6(smi), dtype=torch.float32)
#         if store_lbl:
#             data.y = torch.tensor(df.loc[i, LABEL_COLS].values,
#                                 dtype=torch.float32)        # 5-d vector
#         else:
#             data.y = torch.zeros(5)

#         # 4) hop-distance padding -----------------------------------------
#         dist = hop_distance(data.edge_index, data.x.size(0))
#         pad = torch.full((MAX_NODES, MAX_NODES), MAX_HOPS, dtype=dist.dtype)
#         pad[:dist.size(0), :dist.size(1)] = dist
#         data.dist = pad.to(torch.uint8)

#         # 5) serialise & write --------------------------------------------
#         buf = GraphData.dump(GraphData.pack(data))
#         txn.put(str(gid).encode(), buf)

#         if (i + 1) % 5_000 == 0:
#             txn.commit(); txn = env.begin(write=True)

#     txn.commit(); env.sync()
# finally:                                    # make DB consistent on Ctrl-C
#     if env.open_db():
#         env.close()

# sz = os.path.getsize(LMDB_OUT) / (1<<20)
# print(f"✓ finished {SPLIT}:  {LMDB_OUT}  ({sz:,.1f} MB)")
