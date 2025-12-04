# metric_model.py
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


class AttentiveSetPool(nn.Module):
    """
    Lightweight attention pooling over a set of vectors (B, N, D) -> (B, D),
    optionally mask-aware.
    """
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.q = nn.Linear(dim, hidden, bias=True)
        self.k = nn.Linear(dim, hidden, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.scale = hidden ** 0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    (B, N, D)
        mask: (B, N) bool or 0/1, True/1 for valid positions
        """
        q = self.q(x)  # (B,N,H)
        k = self.k(x)  # (B,N,H)
        v = self.v(x)  # (B,N,D)

        scores = (q * k).sum(dim=-1) / self.scale  # (B,N)
        if mask is not None:
            # mask out invalid positions
            scores = scores.masked_fill(~mask.bool(), -1e9)

        attn = F.softmax(scores, dim=1).unsqueeze(-1)  # (B,N,1)
        out = (attn * v).sum(dim=1)  # (B,D)
        return out


class EntryEncoder(nn.Module):
    """
    Encodes a variable-length set of visual features plus (optional) attribute vector
    into a normalized embedding.

    Expected inputs:
        vis:  (B, N, Dv)
        attr: (B, Da) or None / empty tensor
        mask: (B, N) bool or 0/1, True/1 where vis is valid
    """
    def __init__(
        self,
        vis_dim: int,
        attr_dim: int,
        emb_dim: int = 128,
        use_attention: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.vis_pool = AttentiveSetPool(vis_dim, hidden=min(128, vis_dim))
        else:
            self.vis_pool = None

        self.vis_proj = nn.Linear(vis_dim, emb_dim)

        self.attr_proj: Optional[nn.Linear]
        if attr_dim and attr_dim > 0:
            self.attr_proj = nn.Linear(attr_dim, emb_dim)
            fuse_in = emb_dim * 2
        else:
            self.attr_proj = None
            fuse_in = emb_dim

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(
        self,
        vis: torch.Tensor,
        attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        vis:  (B, N, Dv) or (N, Dv) -> will be reshaped to (1, N, Dv)
        attr: (B, Da) or (Da,) or empty
        mask: (B, N) or (N,) or None
        """
        if vis.dim() == 2:
            vis = vis.unsqueeze(0)  # (1,N,Dv)
        B, N, Dv = vis.shape

        if mask is None:
            mask = torch.ones((B, N), dtype=torch.bool, device=vis.device)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(dtype=torch.bool, device=vis.device)

        # pool over set
        if self.use_attention and self.vis_pool is not None:
            v = self.vis_pool(vis, mask=mask)  # (B,Dv)
        else:
            m = mask.float().unsqueeze(-1)  # (B,N,1)
            summed = (vis * m).sum(dim=1)
            counts = m.sum(dim=1).clamp_min(1.0)
            v = summed / counts  # (B,Dv)

        v = self.vis_proj(v)  # (B,emb_dim)

        if self.attr_proj is not None and attr is not None and attr.numel() > 0:
            if attr.dim() == 1:
                attr = attr.unsqueeze(0)
            attr = attr.to(v.device)
            a = self.attr_proj(attr)
            x = torch.cat([v, a], dim=-1)
        else:
            x = v

        z = self.fuse(x)
        return l2norm(z)

class CrossAttentionEntryEncoder(nn.Module):
    """
    Multi-frame visual + attribute encoder with attr→vis cross-attention.

    Expected inputs:
        vis:  (B, N, Dv)
        attr: (B, Da) or None / empty tensor
        mask: (B, N) bool or 0/1, True/1 where vis is valid

    This class is API-compatible with EntryEncoder so it can be loaded
    from checkpoints that store:
      - vis_dim, attr_dim, emb_dim, use_attention, dropout
    """

    def __init__(
        self,
        vis_dim: int,
        attr_dim: int,
        emb_dim: int = 128,
        use_attention: bool = False,   # kept for ckpt compat; not used here
        dropout: float = 0.1,
        attn_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.vis_dim = vis_dim
        self.attr_dim = attr_dim
        self.emb_dim = emb_dim

        # Project frame embeddings and attrs into a common space
        self.vis_proj_in = nn.Linear(vis_dim, emb_dim)
        self.attr_proj_in = nn.Linear(attr_dim, emb_dim) if attr_dim > 0 else None

        # Cross-attention: attrs (query) over frames (keys/values)
        # q from attrs, k/v from frames
        attn_hidden = int(attn_hidden)
        self.q_attr = nn.Linear(emb_dim, attn_hidden, bias=False)
        self.k_vis  = nn.Linear(emb_dim, attn_hidden, bias=False)
        self.v_vis  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scale = math.sqrt(float(attn_hidden))

        # Fusion MLP (same pattern as EntryEncoder)
        fuse_in = emb_dim * 2 if attr_dim > 0 else emb_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(
        self,
        vis: torch.Tensor,
        attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # vis: (B, N, Dv) or (N, Dv)
        if vis.dim() == 2:
            vis = vis.unsqueeze(0)  # (1, N, Dv)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        B, N, Dv = vis.shape

        if mask is not None:
            mask = mask.to(vis.device).bool()

        # Project frames into emb_dim
        vis = self.vis_proj_in(vis)  # (B, N, emb_dim)

        if self.attr_proj_in is not None and attr is not None and attr.numel() > 0:
            # Project attrs
            if attr.dim() == 1:
                attr = attr.unsqueeze(0)
            attr = attr.to(vis.device)
            attr_emb = self.attr_proj_in(attr)  # (B, emb_dim)

            # Cross-attention: attr_emb as query over frames
            q = self.q_attr(attr_emb).unsqueeze(1)  # (B, 1, H)
            k = self.k_vis(vis)                     # (B, N, H)
            v = self.v_vis(vis)                     # (B, N, emb_dim)

            # scores: (B, 1, N)
            scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

            attn = F.softmax(scores, dim=-1)                  # (B, 1, N)
            pooled_vis = torch.matmul(attn, v).squeeze(1)     # (B, emb_dim)

            fused_in = torch.cat([pooled_vis, attr_emb], dim=-1)  # (B, 2*emb_dim)
        else:
            # Fallback: no attrs → mean-pool projected frames
            if mask is None:
                pooled_vis = vis.mean(dim=1)                    # (B, emb_dim)
            else:
                m = mask.float().unsqueeze(-1)                  # (B, N, 1)
                summed = (vis * m).sum(dim=1)                   # (B, emb_dim)
                counts = m.sum(dim=1).clamp_min(1.0)            # (B, 1)
                pooled_vis = summed / counts                    # (B, emb_dim)

            fused_in = pooled_vis                               # (B, emb_dim)

        z = self.fuse(fused_in)
        return l2norm(z)

        # Normalize input shapes
        if vis.dim() == 2:
            vis = vis.unsqueeze(0)  # (1,N,Dv)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
        B, N, Dv = vis.shape

        if mask is not None:
            mask = mask.to(vis.device).bool()

        # Project frames
        vis = self.vis_proj_in(vis)  # (B,N,emb_dim)

        # Project attrs (if provided)
        if self.attr_proj_in is not None and attr is not None and attr.numel() > 0:
            if attr.dim() == 1:
                attr = attr.unsqueeze(0)
            attr = attr.to(vis.device)
            attr_emb = self.attr_proj_in(attr)  # (B,emb_dim)

            # Cross-attention: attr_emb as query over frames
            q = self.q_attr(attr_emb).unsqueeze(1)  # (B,1,H)
            k = self.k_vis(vis)                     # (B,N,H)
            v = self.v_vis(vis)                     # (B,N,emb_dim)

            # scores: (B,1,N)
            scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

            attn = F.softmax(scores, dim=-1)       # (B,1,N)
            pooled_vis = torch.matmul(attn, v).squeeze(1)  # (B,emb_dim)

            fused_in = torch.cat([pooled_vis, attr_emb], dim=-1)  # (B,2*emb_dim)

        else:
            # Fallback: no attrs → mean-pool frames
            if mask is None:
                pooled_vis = vis.mean(dim=1)
            else:
                m = mask.float().unsqueeze(-1)           # (B,N,1)
                summed = (vis * m).sum(dim=1)            # (B,emb_dim)
                counts = m.sum(dim=1).clamp_min(1.0)     # (B,1)
                pooled_vis = summed / counts             # (B,emb_dim)

            fused_in = pooled_vis  # (B,emb_dim)

        z = self.fuse(fused_in)
        return l2norm(z)