# # polymer_model_simple.py - Fixed version that handles dimension mismatches

# import torch

# import torch.nn as nn



# class PolymerPredictor(nn.Module):

#    def __init__(self, backbone_ckpt, n_out=5, freeze=True, use_gap=False):

#        super().__init__()

      

#        # Import the backbone class here to avoid import issues

#        import sys

#        sys.path.append('/kaggle/input/polymer')

#        from hybrid_backbone import GNN_Transformer_Hybrid

      

#        self.backbone = GNN_Transformer_Hybrid(

#                gnn_dim=512, hidden_dim=256, rdkit_dim=6,

#                extra_atom_dim=5, dropout_rate=0.2142, activation='GELU')

#        self.backbone.load_state_dict(

#                torch.load(backbone_ckpt, map_location='cpu'))

      

#        if freeze:

#            for p in self.backbone.parameters():

#                p.requires_grad_(False)



#        # Store the original backbone forward method

#        self.use_gap = use_gap

      

#        # New 5-output head - simpler version

#        self.head = nn.Sequential(

#            nn.Linear(512 + 6, 256),    # 512 from backbone + 6 RDKit features

#            nn.GELU(),

#            nn.Linear(256, n_out)

#        )



#    def forward(self, data):

#        # Manually implement backbone forward to avoid dimension issues

#        x = self.backbone.atom_encoder(data.x, data.extra_atom_feats)

#        e = self.backbone.bond_encoder(data.edge_attr)

#        src, dst = data.edge_index



#        # GNN layers

#        for conv, upd in [(self.backbone.conv1, self.backbone.edge_upd1),

#                         (self.backbone.conv2, self.backbone.edge_upd2)]:

#            x = conv(x, data.edge_index, e)

#            e = upd(e, x[src], x[dst])

#            if hasattr(data, 'has_xyz') and data.has_xyz.any():

#                e = self.backbone.triplet(data.pos, data.edge_index, e)



#        # Pack for transformer

#        from torch_geometric.utils import to_dense_batch

#        B = int(data.batch.max()) + 1

#        parts, sizes = [], []

#        for i in range(B):

#            idx = (data.batch == i).nonzero(as_tuple=True)[0]

#            xi = x[idx]

#            parts.append(torch.cat([xi, self.backbone.cls_token], 0))

#            sizes.append(xi.size(0) + 1)



#        all_feat = torch.cat(parts, 0)

#        batch = torch.repeat_interleave(

#            torch.arange(B, device=x.device),

#            torch.tensor(sizes, device=x.device)

#        )

#        pad, mask = to_dense_batch(all_feat, batch)



#        # Transformer without attention bias (simplified)

#        pad_out = self.backbone.transformer(pad, (~mask).float())

      

#        # Extract CLS token

#        cls_idx = mask.sum(1) - 1

#        cls_out = pad_out[torch.arange(B, device=x.device), cls_idx]

      

#        # Fix dimension mismatch: ensure rdkit_feats is properly batched

#        rdkit_feats = data.rdkit_feats.float()
#     #    B = cls.size(0)

#        if rdkit_feats.dim() == 1:

#            # If 1D, it means we have a single graph - expand to batch

#            rdkit_feats = rdkit_feats.unsqueeze(0).expand(B, -1)

#        elif rdkit_feats.dim() == 2 and rdkit_feats.size(0) != B:

#            # If 2D but wrong batch size, take mean across nodes (fallback)

#            rdkit_feats = rdkit_feats.mean(0).unsqueeze(0).expand(B, -1)

      

#        # Final prediction with fixed dimensions

#        features = torch.cat([cls_out, rdkit_feats], dim=1)

#        return self.head(features)