# metric_model.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)

class AttentiveSetPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.q = nn.Linear(dim, hidden, bias=True)
        self.k = nn.Linear(dim, hidden, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.scale = hidden ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
        q = self.q(x[:, :1, :])                # (B,1,H)
        k = self.k(x)                           # (B,N,H)
        v = self.v(x)                           # (B,N,D)
        logits = torch.matmul(q, k.transpose(1, 2)) * self.scale  # (B,1,N)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = torch.softmax(logits, dim=-1)   # (B,1,N)
        pooled = torch.matmul(attn, v).squeeze(1)  # (B,D)
        return pooled

class EntryEncoder(nn.Module):
    """
    Fuses multi-image visual embeddings + attribute vector into an entry embedding.
    """
    def __init__(self, vis_dim: int, attr_dim: int, emb_dim: int = 256, use_attention: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.vis_pool = AttentiveSetPool(vis_dim, hidden=min(128, max(32, vis_dim//2)))
        else:
            self.vis_pool = None
        self.vis_proj = nn.Linear(vis_dim, emb_dim)
        if attr_dim > 0:
            self.attr_proj = nn.Linear(attr_dim, emb_dim)
        else:
            self.attr_proj = None
        fuse_in = emb_dim + (emb_dim if attr_dim > 0 else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, vis: torch.Tensor, attr: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if vis.dim() == 2:
            vis = vis.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
        if self.use_attention:
            v = self.vis_pool(vis, mask=mask)
        else:
            if mask is None:
                v = vis.mean(dim=1)
            else:
                m = mask.float().unsqueeze(-1)    # (B,N,1)
                s = (vis * m).sum(dim=1)
                c = m.sum(dim=1).clamp_min(1.0)
                v = s / c
        v = self.vis_proj(v)
        if self.attr_proj is not None and attr is not None and attr.numel() > 0:
            if attr.dim() == 1:
                attr = attr.unsqueeze(0)
            a = self.attr_proj(attr)
            x = torch.cat([v, a], dim=-1)
        else:
            x = v
        z = self.fuse(x)
        return l2norm(z)
