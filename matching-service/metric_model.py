# metric_model.py
from typing import Optional, Tuple, Dict

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


class AttentiveSetPool(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.q = nn.Linear(dim, hidden)
        self.k = nn.Linear(dim, hidden)
        self.v = nn.Linear(dim, dim)
        self.scale = hidden ** 0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    (B, N, D)
        mask: (B, N) bool (True = keep) or None
        """
        B, N, D = x.shape
        q = self.q(x[:, 0:1, :])        # (B,1,H) - query from first frame
        k = self.k(x)                   # (B,N,H)
        v = self.v(x)                   # (B,N,D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B,1,N)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).bool(), -1e9)

        attn = F.softmax(scores, dim=-1)        # (B,1,N)
        out = torch.matmul(attn, v).squeeze(1)  # (B,D)
        return out

    def forward_with_weights(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Same as forward, but also returns attention weights per frame.

        Returns:
            pooled: (B, D)
            attn:   (B, N) or None
        """
        B, N, D = x.shape
        q = self.q(x[:, 0:1, :])        # (B,1,H)
        k = self.k(x)                   # (B,N,H)
        v = self.v(x)                   # (B,N,D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B,1,N)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).bool(), -1e9)

        attn = F.softmax(scores, dim=-1)      # (B,1,N)
        pooled = torch.matmul(attn, v).squeeze(1)  # (B,D)
        return pooled, attn.squeeze(1)        # (B,D), (B,N)


class EntryEncoder(nn.Module):
    """
    Encodes a variable-length set of visual features plus (optional) attribute vector
    into a normalized embedding.
    Baseline encoder: AttentiveSetPool (or mean) + separate proj + concat fusion.
    Now with optional debug outputs via return_components=True.

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
        # self.use_attention = use_attention

        # if use_attention:
        #     self.vis_pool = AttentiveSetPool(vis_dim, hidden=min(128, vis_dim))
        # else:
        #     self.vis_pool = None

        # self.vis_proj = nn.Linear(vis_dim, emb_dim)

        # self.attr_proj: Optional[nn.Linear]
        # if attr_dim and attr_dim > 0:
        #     self.attr_proj = nn.Linear(attr_dim, emb_dim)
        #     fuse_in = emb_dim * 2
        # else:
        #     self.attr_proj = None
        #     fuse_in = emb_dim

        # self.fuse = nn.Sequential(
        #     nn.Linear(fuse_in, emb_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(emb_dim, emb_dim),
        # )
        self.vis_dim = vis_dim
        self.attr_dim = attr_dim
        self.emb_dim = emb_dim
        self.use_attention = use_attention

        # Visual pooling
        self.vis_pool = AttentiveSetPool(vis_dim, vis_dim) if use_attention else None

        # Projections
        self.vis_proj = nn.Linear(vis_dim, emb_dim)
        self.attr_proj = nn.Linear(attr_dim, emb_dim) if attr_dim > 0 else None

        # Fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def _encode(
        self,
        vis: torch.Tensor,
        attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Core encoding logic that returns all components for debugging.
        vis:  (B, N, Dv)
        attr: (B, Da) or None
        mask: (B, N) bool or 0/1 (1 = valid)
        """
        # Normalize shapes
        if vis.dim() == 2:
            vis = vis.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        B, N, Dv = vis.shape

        if mask is not None:
            mask = mask.to(vis.device).bool()

        # --- Visual pooling ---
        if self.use_attention and self.vis_pool is not None:
            pooled_vis, frame_attn = self.vis_pool.forward_with_weights(vis, mask=mask)
        else:
            if mask is None:
                pooled_vis = vis.mean(dim=1)         # (B, Dv)
                frame_attn = None
            else:
                m = mask.float()                     # (B, N)
                weights = m / m.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, N)
                pooled_vis = (vis * weights.unsqueeze(-1)).sum(dim=1)    # (B, Dv)
                frame_attn = weights                 # (B, N)

        # --- Project visual ---
        vis_proj = F.relu(self.vis_proj(pooled_vis))     # (B, emb_dim)

        # --- Project attrs ---
        if attr is not None and self.attr_proj is not None:
            if attr.dim() == 1:
                attr = attr.unsqueeze(0)
            attr = attr.to(vis.device)
            attr_proj = F.relu(self.attr_proj(attr))     # (B, emb_dim)
            attr_present = True
        else:
            # Keep same scale as vis_proj; this matches old behavior
            attr_proj = torch.zeros_like(vis_proj)
            attr_present = False

        # --- Fusion ---
        fused_in = torch.cat([vis_proj, attr_proj], dim=-1)  # (B, 2*emb_dim)
        z = self.fuse(fused_in)
        z = l2norm(z)

        return {
            "z": z,                          # final fused embedding (B, emb_dim)
            "vis_pooled": pooled_vis,        # pre-projection pooled visual (B, Dv)
            "vis_proj": vis_proj,            # visual embedding in emb_dim (B, emb_dim)
            "attr_proj": attr_proj,          # attr embedding in emb_dim (B, emb_dim, zero if missing)
            "frame_attn": frame_attn,        # (B, N) or None
            "attr_present": torch.tensor(attr_present, device=vis.device),
        }

    # def forward(
    #     self,
    #     vis: torch.Tensor,
    #     attr: Optional[torch.Tensor] = None,
    #     mask: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """
    #     vis:  (B, N, Dv) or (N, Dv) -> will be reshaped to (1, N, Dv)
    #     attr: (B, Da) or (Da,) or empty
    #     mask: (B, N) or (N,) or None
    #     """
    #     if vis.dim() == 2:
    #         vis = vis.unsqueeze(0)  # (1,N,Dv)
    #     B, N, Dv = vis.shape

    #     if mask is None:
    #         mask = torch.ones((B, N), dtype=torch.bool, device=vis.device)
    #     else:
    #         if mask.dim() == 1:
    #             mask = mask.unsqueeze(0)
    #         mask = mask.to(dtype=torch.bool, device=vis.device)

    #     # pool over set
    #     if self.use_attention and self.vis_pool is not None:
    #         v = self.vis_pool(vis, mask=mask)  # (B,Dv)
    #     else:
    #         m = mask.float().unsqueeze(-1)  # (B,N,1)
    #         summed = (vis * m).sum(dim=1)
    #         counts = m.sum(dim=1).clamp_min(1.0)
    #         v = summed / counts  # (B,Dv)

    #     v = self.vis_proj(v)  # (B,emb_dim)

    #     if self.attr_proj is not None and attr is not None and attr.numel() > 0:
    #         if attr.dim() == 1:
    #             attr = attr.unsqueeze(0)
    #         attr = attr.to(v.device)
    #         a = self.attr_proj(attr)
    #         x = torch.cat([v, a], dim=-1)
    #     else:
    #         x = v

    #     z = self.fuse(x)
    #     return l2norm(z)

    def forward(
        self,
        vis: torch.Tensor,
        attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        """
        Default behavior (= training/inference): return z only.
        Debug mode: `return_components=True` returns a dict from _encode().
        """
        comps = self._encode(vis, attr=attr, mask=mask)
        if return_components:
            return comps
        return comps["z"]

class CrossAttentionEntryEncoder(nn.Module):
    """
    Multi-frame visual + attribute encoder with attr→vis cross-attention.

    Cross-attention fusion:
      - project frames and attrs into emb_dim
      - use attr embedding as query over frame embeddings
      - fuse pooled_vis and attr_emb
    With optional debug outputs via `return_components=True`.

    Expected inputs:
        vis:  (B, N, Dv)
        attr: (B, Da) or None / empty tensor
        mask: (B, N) bool or 0/1, True/1 where vis is valid

    This class is API-compatible with EntryEncoder so it can be loaded
    from checkpoints that store:
      - vis_dim, attr_dim, emb_dim, use_attention, dropout
    """

    # def __init__(
    #     self,
    #     vis_dim: int,
    #     attr_dim: int,
    #     emb_dim: int = 128,
    #     use_attention: bool = False,   # kept for ckpt compat; not used here
    #     dropout: float = 0.1,
    #     attn_hidden: int = 128,
    # ) -> None:
    #     super().__init__()
    #     self.vis_dim = vis_dim
    #     self.attr_dim = attr_dim
    #     self.emb_dim = emb_dim

    #     # Project frame embeddings and attrs into a common space
    #     self.vis_proj_in = nn.Linear(vis_dim, emb_dim)
    #     self.attr_proj_in = nn.Linear(attr_dim, emb_dim) if attr_dim > 0 else None

    #     # Cross-attention: attrs (query) over frames (keys/values)
    #     # q from attrs, k/v from frames
    #     attn_hidden = int(attn_hidden)
    #     self.q_attr = nn.Linear(emb_dim, attn_hidden, bias=False)
    #     self.k_vis  = nn.Linear(emb_dim, attn_hidden, bias=False)
    #     self.v_vis  = nn.Linear(emb_dim, emb_dim, bias=False)
    #     self.scale = math.sqrt(float(attn_hidden))

    #     # Fusion MLP (same pattern as EntryEncoder)
    #     fuse_in = emb_dim * 2 if attr_dim > 0 else emb_dim
    #     self.fuse = nn.Sequential(
    #         nn.Linear(fuse_in, emb_dim),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(dropout),
    #         nn.Linear(emb_dim, emb_dim),
    #     )

    # def forward(
    #     self,
    #     vis: torch.Tensor,
    #     attr: Optional[torch.Tensor] = None,
    #     mask: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     # vis: (B, N, Dv) or (N, Dv)
    #     if vis.dim() == 2:
    #         vis = vis.unsqueeze(0)  # (1, N, Dv)
    #         if mask is not None and mask.dim() == 1:
    #             mask = mask.unsqueeze(0)

    #     B, N, Dv = vis.shape

    #     if mask is not None:
    #         mask = mask.to(vis.device).bool()

    #     # Project frames into emb_dim
    #     vis = self.vis_proj_in(vis)  # (B, N, emb_dim)

    #     if self.attr_proj_in is not None and attr is not None and attr.numel() > 0:
    #         # Project attrs
    #         if attr.dim() == 1:
    #             attr = attr.unsqueeze(0)
    #         attr = attr.to(vis.device)
    #         attr_emb = self.attr_proj_in(attr)  # (B, emb_dim)

    #         # Cross-attention: attr_emb as query over frames
    #         q = self.q_attr(attr_emb).unsqueeze(1)  # (B, 1, H)
    #         k = self.k_vis(vis)                     # (B, N, H)
    #         v = self.v_vis(vis)                     # (B, N, emb_dim)

    #         # scores: (B, 1, N)
    #         scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
    #         if mask is not None:
    #             scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

    #         attn = F.softmax(scores, dim=-1)                  # (B, 1, N)
    #         pooled_vis = torch.matmul(attn, v).squeeze(1)     # (B, emb_dim)

    #         fused_in = torch.cat([pooled_vis, attr_emb], dim=-1)  # (B, 2*emb_dim)
    #     else:
    #         # Fallback: no attrs → mean-pool projected frames
    #         if mask is None:
    #             pooled_vis = vis.mean(dim=1)                    # (B, emb_dim)
    #         else:
    #             m = mask.float().unsqueeze(-1)                  # (B, N, 1)
    #             summed = (vis * m).sum(dim=1)                   # (B, emb_dim)
    #             counts = m.sum(dim=1).clamp_min(1.0)            # (B, 1)
    #             pooled_vis = summed / counts                    # (B, emb_dim)

    #         fused_in = pooled_vis                               # (B, emb_dim)

    #     z = self.fuse(fused_in)
    #     return l2norm(z)

    #     # Normalize input shapes
    #     if vis.dim() == 2:
    #         vis = vis.unsqueeze(0)  # (1,N,Dv)
    #         if mask is not None and mask.dim() == 1:
    #             mask = mask.unsqueeze(0)
    #     B, N, Dv = vis.shape

    #     if mask is not None:
    #         mask = mask.to(vis.device).bool()

    #     # Project frames
    #     vis = self.vis_proj_in(vis)  # (B,N,emb_dim)

    #     # Project attrs (if provided)
    #     if self.attr_proj_in is not None and attr is not None and attr.numel() > 0:
    #         if attr.dim() == 1:
    #             attr = attr.unsqueeze(0)
    #         attr = attr.to(vis.device)
    #         attr_emb = self.attr_proj_in(attr)  # (B,emb_dim)

    #         # Cross-attention: attr_emb as query over frames
    #         q = self.q_attr(attr_emb).unsqueeze(1)  # (B,1,H)
    #         k = self.k_vis(vis)                     # (B,N,H)
    #         v = self.v_vis(vis)                     # (B,N,emb_dim)

    #         # scores: (B,1,N)
    #         scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
    #         if mask is not None:
    #             scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

    #         attn = F.softmax(scores, dim=-1)       # (B,1,N)
    #         pooled_vis = torch.matmul(attn, v).squeeze(1)  # (B,emb_dim)

    #         fused_in = torch.cat([pooled_vis, attr_emb], dim=-1)  # (B,2*emb_dim)

    #     else:
    #         # Fallback: no attrs → mean-pool frames
    #         if mask is None:
    #             pooled_vis = vis.mean(dim=1)
    #         else:
    #             m = mask.float().unsqueeze(-1)           # (B,N,1)
    #             summed = (vis * m).sum(dim=1)            # (B,emb_dim)
    #             counts = m.sum(dim=1).clamp_min(1.0)     # (B,1)
    #             pooled_vis = summed / counts             # (B,emb_dim)

    #         fused_in = pooled_vis  # (B,emb_dim)

    #     z = self.fuse(fused_in)
    #     return l2norm(z)
    def __init__(
        self,
        vis_dim: int,
        attr_dim: int,
        emb_dim: int = 128,
        use_attention: bool = True,  # kept for API consistency; not used directly
        dropout: float = 0.1,
        attn_hidden: int = 128,
        fusion_mode: str = "xattn",
    ) -> None:
        super().__init__()
        self.vis_dim = vis_dim
        self.attr_dim = attr_dim
        self.emb_dim = emb_dim
        self.fusion_mode = fusion_mode

        # Project inputs into emb_dim
        self.vis_proj_in = nn.Linear(vis_dim, emb_dim)
        self.attr_proj_in = nn.Linear(attr_dim, emb_dim) if attr_dim > 0 else None

        # Cross-attention parameters (attr → vis)
        self.q_attr = nn.Linear(emb_dim, attn_hidden, bias=False)
        self.k_vis = nn.Linear(emb_dim, attn_hidden, bias=False)
        self.v_vis = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scale = attn_hidden ** 0.5

        # Fusion MLP
        if fusion_mode in ("concat", "xattn"):
            fuse_in_dim = emb_dim * 2
        elif fusion_mode == "sum":
            fuse_in_dim = emb_dim
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def _encode(
        self,
        vis: torch.Tensor,
        attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        vis:  (B, N, Dv)
        attr: (B, Da) or None
        mask: (B, N) bool or 0/1
        """
        # Normalize shapes
        if vis.dim() == 2:
            vis = vis.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        B, N, Dv = vis.shape

        if mask is not None:
            mask = mask.to(vis.device).bool()

        # Project frames → (B,N,emb_dim)
        vis_proj = self.vis_proj_in(vis)

        # Project attrs (if available) → (B,emb_dim)
        if self.attr_proj_in is not None and attr is not None and attr.numel() > 0:
            if attr.dim() == 1:
                attr = attr.unsqueeze(0)
            attr = attr.to(vis.device)
            attr_emb = self.attr_proj_in(attr)           # (B,emb_dim)
            attr_present = True

            # Cross-attention
            q = self.q_attr(attr_emb).unsqueeze(1)       # (B,1,H)
            k = self.k_vis(vis_proj)                     # (B,N,H)
            v = self.v_vis(vis_proj)                     # (B,N,emb_dim)

            scores = torch.matmul(q, k.transpose(1, 2)) / self.scale  # (B,1,N)
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

            attn = F.softmax(scores, dim=-1)             # (B,1,N)
            pooled_vis = torch.matmul(attn, v).squeeze(1)  # (B,emb_dim)

            if self.fusion_mode in ("concat", "xattn"):
                fused_in = torch.cat([pooled_vis, attr_emb], dim=-1)    # (B,2*emb_dim)
            elif self.fusion_mode == "sum":
                fused_in = pooled_vis + attr_emb                        # (B,emb_dim)

            frame_attn = attn.squeeze(1)    # (B,N)
            raw_scores = scores.squeeze(1)  # (B,N)

        else:
            # No attrs → fall back to mean-pooling frames
            if mask is None:
                pooled_vis = vis_proj.mean(dim=1)         # (B,emb_dim)
                frame_attn = None
                raw_scores = None
            else:
                m = mask.float().unsqueeze(-1)            # (B,N,1)
                summed = (vis_proj * m).sum(dim=1)        # (B,emb_dim)
                counts = m.sum(dim=1).clamp_min(1.0)      # (B,1)
                pooled_vis = summed / counts              # (B,emb_dim)
                frame_attn = None
                raw_scores = None

            # No attrs: feed pooled_vis directly to fuse ⇒ effectively visual-only head
            fused_in = pooled_vis
            attr_emb = torch.zeros_like(pooled_vis)
            attr_present = False

        # Fusion MLP + normalization
        z = self.fuse(fused_in)
        z = l2norm(z)

        return {
            "z": z,                        # (B,emb_dim)
            "vis_proj": vis_proj,          # (B,N,emb_dim)
            "pooled_vis": pooled_vis,      # (B,emb_dim)
            "attr_emb": attr_emb,          # (B,emb_dim) or zeros
            "frame_attn": frame_attn,      # (B,N) or None
            "raw_scores": raw_scores,      # (B,N) or None (pre-softmax)
            "attr_present": torch.tensor(attr_present, device=vis.device),
        }

    def forward(
        self,
        vis: torch.Tensor,
        attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        comps = self._encode(vis, attr=attr, mask=mask)
        if return_components:
            return comps
        return comps["z"]