# graphdata_lmdb.py
# ────────────────────────────────────────────────────────────────
"""
Light-weight utilities to read/write PCQM4Mv2 graphs stored in an LMDB
with LZ4 compression *and* a pre-padded hop-distance matrix.

Nothing here touches CUDA, RDKit, or any heavy global state, so it is
safe to import inside PyTorch DataLoader worker processes on Windows.
"""
import lmdb, pickle, lz4.frame as lz4f, torch, numpy as np
from torch_geometric.data import Data
from torch.utils.data     import Dataset

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
        np_=np
        return dict(
            x=d.x.numpy().astype(np_.int16),
            edge_index=d.edge_index.numpy().astype(np_.int32),
            edge_attr=d.edge_attr.numpy().astype(np_.float32),
            pos=d.pos.numpy().astype(np_.float32),
            extra_atom=d.extra_atom_feats.numpy().astype(np_.float32),
            rdkit=d.rdkit_feats.numpy().astype(np_.float32),
            y=d.y.numpy().astype(np_.float32),
            has_xyz=bool(d.has_xyz.item()),
            dist=d.dist.numpy().astype(np_.uint8)
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

# class LMDBDataset(Dataset):
#     def __init__(self, ids, path):
#         self.ids=list(ids); self.path=path; self.env=None
#     def __len__(self): return len(self.ids)
#     def __getitem__(self,i):
#         if self.env is None:
#             self.env=lmdb.open(self.path,readonly=True,lock=False,readahead=False)
#         with self.env.begin(buffers=True) as txn:
#             buf=txn.get(str(self.ids[i]).encode())
#         return GraphData.dict2data( GraphData.undump(bytes(buf)) )
