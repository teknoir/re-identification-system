# frame_filter.py
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np


def l2norm_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


def filter_frames(
    vis_np: np.ndarray,
    *,
    max_frames: Optional[int] = None,
    drop_last_always: bool = False,
    norm_min: float = 1e-6,
    dup_threshold: float = 0.999,
    outlier_z_thresh: float = -1.7,
    max_outliers: int = 4,
    min_keep: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter per-frame visual embeddings for a single entry.

    Args:
        vis_np: (N, D) float32 array of per-frame embeddings.
        max_frames: if set, cap the number of frames *after* filtering by keeping
                    the earliest frames (common for huge tracks).
        drop_last_always: if True, drop the last frame unconditionally when N > 1.
                          Useful if you know your generator appends a bogus "11th"
                          embedding; you can disable once you've fixed upstream.
        norm_min: minimum L2 norm; frames with smaller norm are dropped.
        dup_threshold: if cosine similarity between two frames >= dup_threshold,
                       treat them as near-duplicates and keep only the first.
        outlier_z_thresh: z-score threshold for dropping low-sim outliers based on
                          similarity to the entry mean. Negative number; e.g. -2.0
                          means "drop frames whose sim is < mean - 2 * std".
        max_outliers: maximum number of frames to drop as outliers per entry.
        min_keep: never drop below this many frames; if filtering would reduce
                  frames below this, we stop dropping outliers.

    Returns:
        vis_filtered: (M, D) filtered embeddings (L2-normalized).
        kept_indices: (M,) indices into vis_np that were kept.
    """
    if vis_np.ndim != 2:
        raise ValueError(f"vis_np must have shape (N, D), got {vis_np.shape}")

    N, D = vis_np.shape
    if N == 0:
        return vis_np, np.arange(0, dtype=np.int64)

    # 0) Copy & L2-normalize
    V = np.asarray(vis_np, dtype=np.float32).copy()
    V = l2norm_np(V)  # (N, D)

    # Track which original indices we're keeping
    kept = np.arange(N, dtype=np.int64)

    # 1) Optional: drop the last frame unconditionally (temporary hack for bogus tail)
    if drop_last_always and V.shape[0] > 1:
        V = V[:-1, :]
        kept = kept[:-1]

    # 2) Drop degenerate frames by norm (should be rare if L2-normalized upstream)
    norms = np.linalg.norm(V, axis=1)
    good_norm_mask = norms > norm_min
    if not np.all(good_norm_mask):
        V = V[good_norm_mask]
        kept = kept[good_norm_mask]

    if V.shape[0] <= 1:
        return V, kept

    # 3) Remove within-entry near-duplicate frames
    #    (keeps the first occurrence, drops later ones)
    M = V.shape[0]
    keep_mask = np.ones(M, dtype=bool)
    for i in range(M):
        if not keep_mask[i]:
            continue
        vi = V[i]
        # compare only to later frames
        sims = V[i + 1 :] @ vi
        dup_idx = np.where(sims >= dup_threshold)[0]  # indices relative to V[i+1:]
        if dup_idx.size > 0:
            keep_mask[i + 1 + dup_idx] = False

    V = V[keep_mask]
    kept = kept[keep_mask]

    if V.shape[0] <= 1:
        return V, kept

    # 4) Drop strong outliers based on similarity to entry mean
    #    (this targets weird crops / multi-person frames)
    M = V.shape[0]
    if M > min_keep and max_outliers > 0:
        mean_vec = l2norm_np(V.mean(axis=0, keepdims=True))[0]  # (D,)
        sims = V @ mean_vec                                   # (M,)

        mu = float(sims.mean())
        sigma = float(sims.std() + 1e-6)
        # lower is worse; z = (sim - mu)/sigma
        z_scores = (sims - mu) / sigma

        # candidates are frames far below the mean (small similarity)
        candidate_idx = np.where(z_scores < outlier_z_thresh)[0]

        if candidate_idx.size > 0:
            # sort by z ascending (most negative first)
            order = np.argsort(z_scores[candidate_idx])
            to_drop = candidate_idx[order][:max_outliers]

            # ensure we don't go below min_keep
            if M - to_drop.size >= min_keep:
                keep_mask2 = np.ones(M, dtype=bool)
                keep_mask2[to_drop] = False
                V = V[keep_mask2]
                kept = kept[keep_mask2]

    # 5) Optional cap on total frames
    if max_frames is not None and V.shape[0] > max_frames:
        V = V[:max_frames, :]
        kept = kept[:max_frames]

    return V, kept
