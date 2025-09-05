# # polymer_model.py ----------------------------------------------------------
# import torch 
# import torch.nn as nn
# from hybrid_backbone import GNN_Transformer_Hybrid     # your existing code

# class PolymerPredictor(nn.Module):
#     def __init__(self, backbone_ckpt, n_out=5, freeze=True, use_gap=False):
#         super().__init__()
#         self.backbone = GNN_Transformer_Hybrid(
#                 gnn_dim=512, hidden_dim=256, rdkit_dim=6,
#                 extra_atom_dim=5, dropout_rate=0.2142, activation='GELU')
#         self.backbone.load_state_dict(
#                 torch.load(backbone_ckpt, map_location='cpu'))
#         if freeze:
#             for p in self.backbone.parameters():
#                 p.requires_grad_(False)

#         # optionally keep the 1-dim gap head
#         self.use_gap = use_gap
#         if use_gap:
#             self.gap_head = self.backbone.mlp          # reuse weights
#             in_dim = 512 + 1
#         else:
#             in_dim = 512

#         # new 5-output head
#         self.head = nn.Sequential(
#             nn.Linear(in_dim + 6, 256),    # +6 for RDKit globals
#             nn.GELU(),
#             nn.Linear(256, n_out)
#         )

#     def forward(self, data):
#         cls = self.backbone.forward_backbone_only(data)  # returns 512-d CLS
        
#         # Handle rdkit_feats dimensions properly
#         rdkit_feats = data.rdkit_feats.float()
#         B = cls.size(0)
#         if rdkit_feats.dim() == 1 and rdkit_feats.numel() == B * 6:
#             rdkit_feats = rdkit_feats.view(B, 6)   # <- fix
#         if rdkit_feats.dim() == 1:
#             # Single molecule case - expand to match batch size
#             batch_size = cls.size(0)
#             rdkit_feats = rdkit_feats.unsqueeze(0).expand(batch_size, -1)
#         elif rdkit_feats.dim() == 2:
#             # Already batched, but might need to aggregate if it's per-atom features
#             if rdkit_feats.size(0) != cls.size(0):
#                 # If rdkit_feats has more rows than batch size, it might be per-atom
#                 # In that case, we need to aggregate by molecule
#                 batch_size = cls.size(0)
#                 rdkit_global = []
#                 for i in range(batch_size):
#                     mask = (data.batch == i)
#                     if mask.any():
#                         rdkit_global.append(rdkit_feats[mask].mean(0))
#                     else:
#                         rdkit_global.append(torch.zeros(6, device=rdkit_feats.device))
#                 rdkit_feats = torch.stack(rdkit_global, dim=0)
        
#         # Debug prints to understand dimensions
#         # print(f"cls shape: {cls.shape}")
#         # print(f"rdkit_feats shape: {rdkit_feats.shape}")
        
#         if self.use_gap:
#             gap_input = torch.cat([cls, rdkit_feats], dim=1)
#             # print(f"gap_input shape: {gap_input.shape}")
#             gap = self.gap_head(gap_input)
#             # print(f"gap shape: {gap.shape}")
#             feats = torch.cat([cls, gap], dim=1)
#             # print(f"feats (with gap) shape: {feats.shape}")
#         else:
#             feats = cls
#             # print(f"feats (no gap) shape: {feats.shape}")
        
#         final_input = torch.cat([feats, rdkit_feats], dim=1)
#         # print(f"final_input shape: {final_input.shape}")
#         # print(f"Expected input size for head: {self.head[0].in_features}")
        
#         out = self.head(final_input)
#         return out

# polymer_model.py ----------------------------------------------------------
import torch 
import torch.nn as nn
from hybrid_backbone import GNN_Transformer_Hybrid
import torch.nn.functional as F

class PolymerPredictor(nn.Module):
    def __init__(self, backbone_ckpt, n_out=5, freeze=True, use_gap=False, hidden=512, rdkit_dim=6):
        super().__init__()
        self.backbone = GNN_Transformer_Hybrid(
                gnn_dim=512, hidden_dim=256, rdkit_dim=6,
                extra_atom_dim=5, dropout_rate=0.2142, activation='GELU')
        self.backbone.load_state_dict(
                torch.load(backbone_ckpt, map_location='cpu'))
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.use_gap = use_gap
        self.hidden = 512
        self.rdkit_dim = 6
        in_dim = self.hidden * 2 if use_gap else self.hidden 
        # new 5-output head
        self.head = nn.Sequential(
            nn.Linear(in_dim + self.rdkit_dim, 256),    # +6 for RDKit globals
            nn.GELU(),
            nn.Linear(256, n_out)
        )

    def forward(self, data):
        if self.use_gap:
            # backbone must support return_gap=True and return (cls, gap)
            cls, g = self.backbone.forward_backbone_only(data, return_gap=True) # [B,512], [B, 512]
            # normalize each vector
            cls = F.layer_norm(cls, (cls.size(-1),))
            g   = F.layer_norm(g, (g.size(-1),))
            feats = torch.cat([cls, g], dim=1) # [B,1024]
        else:
            cls = self.backbone.forward_backbone_only(data) # [B, 512]
            cls = F.layer_norm(cls, (cls.size(-1),))
            feats = cls
        
        # Handle rdkit_feats dimensions properly
        rdkit_feats = data.rdkit_feats.float()
        B = cls.size(0)
        if rdkit_feats.dim() == 1 and rdkit_feats.numel() == B * 6:
            rdkit_feats = rdkit_feats.view(B, 6)   # <- fix
        if rdkit_feats.dim() == 1:
            # Single molecule case - expand to match batch size
            batch_size = cls.size(0)
            rdkit_feats = rdkit_feats.unsqueeze(0).expand(batch_size, -1)
        elif rdkit_feats.dim() == 2:
            # Already batched, but might need to aggregate if it's per-atom features
            if rdkit_feats.size(0) != cls.size(0):
                # If rdkit_feats has more rows than batch size, it might be per-atom
                # In that case, we need to aggregate by molecule
                batch_size = cls.size(0)
                rdkit_global = []
                for i in range(batch_size):
                    mask = (data.batch == i)
                    if mask.any():
                        rdkit_global.append(rdkit_feats[mask].mean(0))
                    else:
                        rdkit_global.append(torch.zeros(6, device=rdkit_feats.device))
                rdkit_feats = torch.stack(rdkit_global, dim=0)
        
        
        # feats set above so append RDKit globals
        final_input = torch.cat([feats, rdkit_feats], dim=1) # [B, in_dim+6]
        
        out = self.head(final_input)
        return out
