# ---------------------------------------------------------- hybrid_backbone.py ----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GINEConv, global_mean_pool
from torch_geometric.utils import add_self_loops, to_dense_batch, to_dense_adj
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import random
import copy

def has_nan(t: torch.Tensor) -> bool:
    """True -> tensor contains NaN or +/- Inf (works on fp16/fp32)"""
    return torch.isfinite(t).logical_not().any().item()

class ExtendedAtomEncoder(nn.Module):
    def __init__(self, gnn_dim, extra_atom_dim):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=gnn_dim)
        self.extra_proj = nn.Linear(extra_atom_dim, gnn_dim)
        self.output_proj = nn.Linear(gnn_dim * 2, gnn_dim) # Combine encoded + extra info

    def forward(self, x, extra_atom_features):
        """
        x: [num_atoms, atom_input_dim] -> original atom input (atomic number idx)
        extra_atom_features: [num_atoms, extra_atom_dim] -> handcrafted features
        """
        atom_emb = self.atom_encoder(x) # [num_atoms, gnn_dim]
        extra_emb = self.extra_proj(extra_atom_features.float()) # [num_atoms, gnn_dim]
        extra_emb = torch.nan_to_num(extra_emb, nan=0.0, posinf=0.0, neginf=0.0)
        combined = torch.cat([atom_emb, extra_emb], dim=-1) # [num_atoms, gnn_dim*2]
        output = self.output_proj(combined) # [num_atoms, gnn_dim]
        return output

class BondEncoderWithDist(nn.Module):
    """
    First three cols = OGB categorical bond features (int)
    Remaining K cols = continuous geometry embedding (float)
    """
    def __init__(self, emb_dim: int, n_rbf: int = 32):
        super().__init__()
        self.cat_enc = BondEncoder(emb_dim) # (E,3)  --> (E,D)
        self.dist_mlp = nn.Sequential(      # (E,K)  --> (E,D)
            nn.Linear(n_rbf, emb_dim), 
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
            )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        cat = self.cat_enc(edge_attr[:, :3].long())
        geo = self.dist_mlp(edge_attr[:, 3:].float())
        return cat + geo

class PairWiseUpdate(nn.Module):
    """
    Edge updater e_ij <-- f(e_ij , h_i , h_j)
    Keeps the size (E, D) unchanged so batching still works
    """
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(3*dim, dim), 
            nn.SiLU(),
            nn.Linear(dim, dim)
            )
    
    def forward(self, e, hi, hj): # all (E, D)
        return self.f(torch.cat([e, hi, hj], -1))

class TripletBlock(nn.Module):
    def __init__(self, dim, num_samples=4):
        super().__init__()
        self.angle_mlp = nn.Sequential(
            nn.Linear(1, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim)
            )
        self.num_samples = num_samples

    def forward(self, pos, edge_index, e): # pos: (N,3), edge_index: (2,E), e: (E,D)
        device = pos.device
        src, dst = edge_index[0], edge_index[1]
        num_nodes = pos.size(0)

        # Build adjacency list
        adj = [[] for _ in range(num_nodes)]
        for i, j in zip(src.tolist(), dst.tolist()):
            adj[i].append(j)

        # Sample k for each edge (i, j)
        new_i, new_j, new_k, group_idx = [], [], [], []
        for idx, (i, j) in enumerate(zip(src.tolist(), dst.tolist())):
            neighbors = [k for k in adj[i] if k != j]
            if len(neighbors) == 0:
                continue
            ks = random.sample(neighbors, min(self.num_samples, len(neighbors)))
            for k in ks:
                new_i.append(i)
                new_j.append(j)
                new_k.append(k)
                group_idx.append(idx) # for aggregation

        if len(new_i) == 0:
            return e  # no triplets found

        i = torch.tensor(new_i, device=device)
        j = torch.tensor(new_j, device=device)
        k = torch.tensor(new_k, device=device)
        group_idx = torch.tensor(group_idx, device=device)

        v1 = pos[i] - pos[j]
        v2 = pos[i] - pos[k]
        cos = (v1 * v2).sum(-1, keepdim=True) / (1.norm(dim=-1, keepdim=True) * v2.norm(dim=-1, keepdim=True) + 1e-9)
        ang = torch.acos(torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)) # (T, 1)
        gate = self.angle_mlp(ang) # (T, D)

        # Aggregate angle gates over original edges
        from torch_scatter import scatter
        angle_update = scatter(gate, group_idx, dim=0, dim_size=e.size(0), reduce='mean')
        return e + angle_update

class TransformerEncoderLayerWithBias(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,
                 dim_feedforward=2048, activation="ReLU", batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "ReLU" else F.gelu
        self.nhead = nhead  # keep for external checks

    def forward(self, src, src_key_padding_mask=None, attn_bias=None):
        B, L, _ = src.size()

        # Build attention mask
        # expected shape: (B · nhead,  L,  L)
        if attn_bias is not None:
            assert attn_bias.shape == (B, self.nhead, L, L), f"attn_bias must be (B,H,L,L), got {attn_bias.shape}"
            attn_mask = attn_bias.view(B * self.nhead, L, L)
        else:
            attn_mask = None

        # MultiheadAttention already projects and splits into heads internally
        attn_output, _ = self.self_attn(src, src, src, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src, mask, attn_bias=None):
        for mod in self.layers:
            src = mod(src, src_key_padding_mask=mask, attn_bias=attn_bias)
        return self.norm(src)

class GNN_Transformer_Hybrid(nn.Module):
    def __init__(self, gnn_dim, rdkit_dim, hidden_dim, extra_atom_dim, 
                 num_transformer_layers=2, num_heads=8, dropout_rate=0.2, activation='GELU'):
        super().__init__()
        act_map = {
            'ReLU': nn.ReLU(), 
            'ELU': nn.ELU(), 
            'LeakyReLU': nn.LeakyReLU(), 
            'PReLU': nn.PReLU(), 
            'GELU': nn.GELU(), 
            'Swish': nn.SiLU()
            }
        act_fn = act_map[activation]
        self.gnn_dim = gnn_dim
        self.rdkit_dim = rdkit_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.atom_encoder = ExtendedAtomEncoder(gnn_dim, extra_atom_dim)
        self.bond_encoder = BondEncoderWithDist(gnn_dim, n_rbf=32)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, gnn_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Message passing
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(gnn_dim, gnn_dim), act_fn, nn.Linear(gnn_dim, gnn_dim)))
        self.edge_upd1 = PairWiseUpdate(gnn_dim)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(gnn_dim, gnn_dim), act_fn, nn.Linear(gnn_dim, gnn_dim)))
        self.edge_upd2 = PairWiseUpdate(gnn_dim)
        self.triplet = TripletBlock(gnn_dim)

        # Custom Transformer encoder
        encoder_layer = TransformerEncoderLayerWithBias(
            d_model=gnn_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=gnn_dim * 4,
            dropout=dropout_rate
        )
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Final prediction head
        self.mlp = nn.Sequential(
            nn.Linear(gnn_dim + rdkit_dim, hidden_dim), act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2), act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        # 1) Encoders
        x = self.atom_encoder(data.x, data.extra_atom_feats)
        if has_nan(x):
            raise RuntimeError("NaN after atom_encoder")

        e = self.bond_encoder(data.edge_attr)
        src, dst = data.edge_index

        # GNN layers with edge updates
        for conv, upd in [(self.conv1, self.edge_upd1), (self.conv2, self.edge_upd2)]:
            x = conv(x, data.edge_index, e)
            e = upd(e, x[src], x[dst])
            if data.has_xyz.any():
                e = self.triplet(data.pos, data.edge_index, e)
        if has_nan(x):
            raise RuntimeError("NaN after GINE")

        # 2) Pack graphs (+CLS)
        B = int(data.batch.max()) + 1
        parts, sizes = [], []
        for i in range(B):
            idx = (data.batch == i).nonzero(as_tuple=True)[0]
            xi = x[idx]
            parts.append(torch.cat([xi, self.cls_token], 0))
            sizes.append(xi.size(0) + 1)

        all_feat = torch.cat(parts, 0)
        batch = torch.repeat_interleave(
            torch.arange(B, device=x.device),
            torch.tensor(sizes, device=x.device)
        )
        pad, mask = to_dense_batch(all_feat, batch) # (B,L,D)
        max_nodes = pad.size(1)

        # 3) Attention bias (fp32, B × H × L × L)
        bias_stack = []
        dist_iter = iter(data.dist) # (MAX_NODES, MAX_NODES)

        for i in range(B):
            dist = next(dist_iter).to(x.device)  # uint8 -> device
            n = sizes[i] - 1 # real atoms (–CLS)

            # Safety-net, empty graph
            if n == 0:
                bias_stack.append(torch.zeros(max_nodes, max_nodes,
                                           dtype=torch.float32,
                                           device=x.device))
                continue

            # Crop and negate distances, then CLAMP
            local = -dist[:n, :n].float() # 0 on diag, –d elsewhere
            tmp = torch.zeros(max_nodes, max_nodes,
                            dtype=torch.float32,
                            device=x.device)
            tmp[:n, :n] = local.clamp(min=-5.0, max=0.0) # Safety-net
            bias_stack.append(tmp)

        attn_bias = torch.stack(bias_stack, 0) # (B,L,L)
        attn_bias = attn_bias.unsqueeze(1) # (B,1,L,L)
        attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1) # (B,H,L,L)

        if has_nan(attn_bias):
            raise RuntimeError("NaN in attention-bias")

        # 4) Transformer
        pad_fp32 = pad.float() # (B,L,D) fp32 -- safer numerics
        valid_mask = mask # (B,L) bool True==real node
        key_padding = (~valid_mask).float() # (B,L) fp32 1==PAD, 0==keep

        # Forward through the stacked encoder layers
        pad_out = self.transformer(
            pad_fp32, # src
            key_padding, # src_key_padding_mask (fp32)
            attn_bias # attn_bias (fp32, B×H×L×L)

            )  
        # Last ditch effort to sanitize to replace any NaNs/Infs
        if has_nan(pad_out):
            pad_out = torch.nan_to_num(pad_out, nan=0.0, posinf=0.0, neginf=0.0)

        # CLS token is always the last real position in each sequence
        cls_idx = valid_mask.sum(1) - 1 # (B,)
        cls_out = pad_out[torch.arange(B, device=x.device), cls_idx]

        # Final prediction
        out = torch.cat([cls_out, data.rdkit_feats.float()], dim=1)
        return self.mlp(out)
    
    def forward_backbone_only(self, data, return_gap: bool = False):
        """Forward pass through backbone only.
        Returns CLS token output (default) and, if `return_gap=True`, also returns GAP over atom embeddings.
        """
        # 1) Encoders
        x = self.atom_encoder(data.x, data.extra_atom_feats)
        if has_nan(x):
            raise RuntimeError("NaN after atom_encoder")

        e = self.bond_encoder(data.edge_attr)
        src, dst = data.edge_index

        # GNN layers with edge updates
        for conv, upd in [(self.conv1, self.edge_upd1), (self.conv2, self.edge_upd2)]:
            x = conv(x, data.edge_index, e)
            e = upd(e, x[src], x[dst])
            if data.has_xyz.any():
                e = self.triplet(data.pos, data.edge_index, e)
        if has_nan(x):
            raise RuntimeError("NaN after GINE")

        # 2) Pack graphs (+CLS)
        B = int(data.batch.max()) + 1
        parts, sizes = [], []
        for i in range(B):
            idx = (data.batch == i).nonzero(as_tuple=True)[0]
            xi = x[idx] # [n_i, D]
            parts.append(torch.cat([xi, self.cls_token], 0)) # [n_i+1, D] (CLS last)
            sizes.append(xi.size(0) + 1)

        all_feat = torch.cat(parts, 0) # [sum(n_i)+B, D]
        batch = torch.repeat_interleave(
            torch.arange(B, device=x.device),
            torch.tensor(sizes, device=x.device)
        )
        pad, mask = to_dense_batch(all_feat, batch) # pad: [B, L, D], mask: [B, L] (True = real)

        max_nodes = pad.size(1)

        # 3) Attention bias (fp32, B × H × L × L)
        bias_stack = []
        dist_iter = iter(data.dist) # (MAX_NODES, MAX_NODES)

        for i in range(B):
            dist = next(dist_iter).to(x.device) # uint8 -> device
            n = sizes[i] - 1 # real atoms (–CLS)

            # Safety-net empty graph
            if n == 0:
                bias_stack.append(torch.zeros(max_nodes, max_nodes, dtype=torch.float32, device=x.device))
                continue

            # Crop and negate distances, then CLAMP
            local = -dist[:n, :n].float() # 0 on diag, –d elsewhere
            tmp = torch.zeros(max_nodes, max_nodes,
                            dtype=torch.float32,
                            device=x.device)
            tmp[:n, :n] = local.clamp(min=-5.0, max=0.0) # Safety-net 
            bias_stack.append(tmp)

        attn_bias = torch.stack(bias_stack, 0) # (B, L, L)
        attn_bias = attn_bias.unsqueeze(1) # (B, 1, L, L)
        attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1) # (B, H, L, L)

        if has_nan(attn_bias):
            raise RuntimeError("NaN in attention-bias")

        # 4) Transformer
        pad_fp32 = pad.float()  # (B, L, D) fp32 for safer numerics
        valid_mask = mask # (B, L) bool True==real (includes CLS)
        key_padding = (~valid_mask).float() # (B, L) fp32 1==PAD, 0==keep
        pad_out = self.transformer(pad_fp32, key_padding, attn_bias) # (B, L, D)
        # sanitizer to replace any NaNs/Infs
        if has_nan(pad_out):
            pad_out = torch.nan_to_num(pad_out, nan=0.0, posinf=0.0, neginf=0.0)

        # CLS token is the last real position in each sequence
        cls_idx = valid_mask.sum(1) - 1 # (B,)
        cls_out = pad_out[torch.arange(B, device=x.device), cls_idx] # (B, D)

        if not return_gap:
            return cls_out

        # True GAP over atom positions (exclude CLS) 
        # Build a mask that is True only for atom tokens (positions < cls_idx)
        L = pad_out.size(1)
        sizes_t = torch.tensor(sizes, device=pad_out.device) # [B]
        pos = torch.arange(L, device=pad_out.device).unsqueeze(0).expand(B, L) # [B, L]
        atom_mask = pos < (sizes_t - 1).unsqueeze(1) # [B, L], True for atoms only
        # (valid_mask is already True for real tokens; atoms are both real and not-CLS)
        atom_mask = atom_mask & valid_mask
        atoms_sum = (pad_out * atom_mask.unsqueeze(-1)).sum(dim=1) # [B, D]
        denom = atom_mask.sum(dim=1).clamp_min(1).unsqueeze(1).to(pad_out.dtype) # [B, 1]
        gap_out = atoms_sum / denom # [B, D]

        return cls_out, gap_out