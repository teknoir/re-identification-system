# metric_model.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.q = nn.Linear(in_dim, hidden, bias=False)
        self.k = nn.Linear(in_dim, hidden, bias=False)
        self.v = nn.Linear(in_dim, hidden, bias=False)
        self.o = nn.Linear(hidden, in_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T)
        B, T, D = x.shape
        q = self.q(x)  # (B, T, H)
        k = self.k(x)  # (B, T, H)
        v = self.v(x)  # (B, T, H)
        scores = torch.einsum("bth,bTh->btT", q, k) / (q.shape[-1] ** 0.5)  # (B, T, T)
        # mask: invalid positions = -inf
        attn_mask = (~mask).unsqueeze(1).expand(-1, T, -1)  # (B, T, T)
        scores = scores.masked_fill(attn_mask, float("-inf"))
        A = torch.softmax(scores, dim=-1)  # (B, T, T)
        ctx = torch.einsum("btT,bTh->bth", A, v)  # (B, T, H)
        out = self.o(ctx)  # (B, T, D)
        # pool across time using mask
        out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        pooled = out.sum(dim=1, keepdim=False) / denom.squeeze(-1)  # (B, D)
        return pooled

class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return x.sum(dim=1) / denom

class EntryEncoder(nn.Module):
    def __init__(self, vis_dim: int, attr_dim: int, emb_dim: int, use_attention: bool = True, dropout: float = 0.0):
        super().__init__()
        self.vis_dim = vis_dim
        self.attr_dim = attr_dim
        self.emb_dim = emb_dim
        self.use_attention = use_attention

        self.temporal = TemporalAttention(vis_dim) if use_attention else MeanPool()
        self.attr_mlp = nn.Sequential(
            nn.Linear(attr_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(vis_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, emb_dim),
        )

    def forward(self, vis: torch.Tensor, mask: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        # vis: (B, T, Dv), mask: (B, T), attr: (B, Da)
        v = self.temporal(vis, mask)            # (B, Dv)
        a = self.attr_mlp(attr)                 # (B, 256)
        z = self.fuse(torch.cat([v, a], dim=-1))  # (B, emb)
        z = F.normalize(z, p=2, dim=-1)         # L2-normalize
        return z

class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (InfoNCE style), uses cosine similarity
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: (B, D) L2-normalized, y: (B,)
        sim = z @ z.t()  # cosine because z normalized
        B = z.shape[0]
        mask_pos = (y.unsqueeze(1) == y.unsqueeze(0)) & (~torch.eye(B, dtype=torch.bool, device=z.device))
        mask_neg = ~mask_pos & (~torch.eye(B, dtype=torch.bool, device=z.device))

        # For stability, subtract max
        logits = sim / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values

        # For each i, positives are those j where mask_pos[i,j]==1
        exp_logits = torch.exp(logits) * (mask_pos | mask_neg)  # zero out diagonal
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Only average over positives
        pos_log_prob = (log_prob * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1).clamp(min=1))
        loss = -pos_log_prob.mean()
        return loss

def batch_pos_neg_stats(z: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    sim = z @ z.t()
    B = z.shape[0]
    mask_pos = (y.unsqueeze(1) == y.unsqueeze(0)) & (~torch.eye(B, dtype=torch.bool, device=z.device))
    mask_neg = ~mask_pos & (~torch.eye(B, dtype=torch.bool, device=z.device))
    pos = sim[mask_pos]
    neg = sim[mask_neg]
    pos_m = float(pos.mean().item()) if pos.numel() else 0.0
    neg_m = float(neg.mean().item()) if neg.numel() else 0.0
    return pos_m, neg_m


def _extract_meta(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if "meta" in ckpt:
        return dict(ckpt["meta"])
    required = ["vis_dim", "attr_dim", "emb_dim"]
    for key in required:
        if key not in ckpt:
            raise ValueError(f"Checkpoint missing required key '{key}'")
    return {
        "vis_dim": ckpt["vis_dim"],
        "attr_dim": ckpt["attr_dim"],
        "emb_dim": ckpt["emb_dim"],
        "use_attention": ckpt.get("use_attention", True),
        "dropout": ckpt.get("dropout", 0.0),
    }


def load_entry_encoder(ckpt_path: str, device: torch.device | str = "cpu") -> Tuple[EntryEncoder, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = _extract_meta(ckpt)
    state_dict = ckpt.get("state_dict") or ckpt.get("model")
    if state_dict is None:
        raise ValueError("Checkpoint must contain 'state_dict' (or legacy 'model') weights.")
    model = EntryEncoder(
        meta["vis_dim"],
        meta["attr_dim"],
        meta["emb_dim"],
        use_attention=meta.get("use_attention", True),
        dropout=meta.get("dropout", 0.0),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model, meta


@torch.no_grad()
def encode_numpy(
    model: EntryEncoder,
    vis_np: np.ndarray,
    mask_np: Optional[np.ndarray] = None,
    attr_np: Optional[np.ndarray] = None,
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """
    Convenience helper to run the encoder on numpy arrays.
    Args:
        vis_np:  np.ndarray shaped (T, vis_dim) or (1, T, vis_dim)
        mask_np: optional boolean mask (T,) or (1, T)
        attr_np: optional attr vector (attr_dim,) or (1, attr_dim)
    Returns:
        L2-normalized embedding as np.ndarray (emb_dim,)
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    vis_t = torch.from_numpy(vis_np).float()
    if vis_t.dim() == 2:
        vis_t = vis_t.unsqueeze(0)
    if vis_t.dim() != 3:
        raise ValueError("vis_np must be rank-2 (T,D) or rank-3 (B,T,D)")
    mask_arr = mask_np
    if mask_arr is None:
        mask_arr = np.ones((vis_t.shape[0], vis_t.shape[1]), dtype=bool)
    mask_t = torch.from_numpy(mask_arr.astype(bool))
    if mask_t.dim() == 1:
        mask_t = mask_t.unsqueeze(0)
    attr_arr = attr_np
    attr_dim = model.attr_dim
    if attr_dim == 0:
        attr_t = torch.zeros((vis_t.shape[0], 0), dtype=torch.float32)
    else:
        if attr_arr is None or attr_arr.size == 0:
            attr_arr = np.zeros((vis_t.shape[0], attr_dim), dtype=np.float32)
        attr_t = torch.from_numpy(attr_arr.astype(np.float32, copy=False))
        if attr_t.dim() == 1:
            attr_t = attr_t.unsqueeze(0)
    vis_t = vis_t.to(dev)
    mask_t = mask_t.to(dev)
    attr_t = attr_t.to(dev)
    out = model(vis_t, mask_t, attr_t)
    return out.squeeze(0).detach().cpu().numpy()
