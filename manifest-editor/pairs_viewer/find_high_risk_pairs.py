#!/usr/bin/env python
"""
find_high_risk_pairs.py

Offline tool to surface potentially mislabeled pairs in the GT:

- "Hard negatives": different person_id but *high* cosine similarity.
- "Hard positives": same person_id but *low* cosine similarity.

Input:
  - entry_vectors.npy: fused entry embeddings (L2-normalized).
  - entry_ids.txt: list of entry_ids (same order as rows in entry_vectors.npy).
  - manifest.json: entry_id -> payload (images, store_id, day_id, camera, timestamp, attrs, ...).
  - multi_gt.json: entry_id -> person_id.

Output:
  - pairs_hard.jsonl: one JSON object per candidate pair.
  - pairs_hard.csv: for quick spreadsheet triage.

Usage:
  python find_high_risk_pairs.py \
      --vecs entry_vectors.npy \
      --ids entry_ids.txt \
      --manifest manifest.json \
      --gt multi_gt.json \
      --output-dir ./pairs \
      --neg-threshold 0.88 \
      --pos-threshold 0.75 \
      --topk-neg 20
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


@dataclass
class PairRecord:
    entry_id_a: str
    entry_id_b: str
    pid_a: str
    pid_b: str
    same_pid: int
    sim: float
    risk_type: str  # "hard_neg", "hard_pos", or "alias_candidate"
    # basic meta
    store_a: Optional[str] = None
    store_b: Optional[str] = None
    day_a: Optional[str] = None
    day_b: Optional[str] = None
    camera_a: Optional[str] = None
    camera_b: Optional[str] = None
    ts_a: Optional[str] = None
    ts_b: Optional[str] = None
    # first image URIs (for quick preview / UI)
    image_a: Optional[str] = None
    image_b: Optional[str] = None

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def load_embeddings(vec_path: Path, ids_path: Path) -> Tuple[np.ndarray, List[str]]:
    V = np.load(vec_path).astype(np.float32)
    # L2 normalize just in case
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    ids = [s.strip() for s in ids_path.read_text().splitlines() if s.strip()]
    if len(ids) != V.shape[0]:
        raise ValueError(f"Got {len(ids)} ids but {V.shape[0]} vectors")
    return V, ids


def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    return json.loads(manifest_path.read_text())


def load_gt(gt_path: Path) -> Dict[str, str]:
    return json.loads(gt_path.read_text())


def build_pid_index(ids: List[str], entry_to_pid: Dict[str, str]) -> Tuple[List[Optional[str]], Dict[str, List[int]]]:
    pid_of: List[Optional[str]] = []
    idxs_by_pid: Dict[str, List[int]] = {}
    for i, eid in enumerate(ids):
        pid = entry_to_pid.get(eid)
        pid_of.append(pid)
        if pid is not None:
            idxs_by_pid.setdefault(pid, []).append(i)
    return pid_of, idxs_by_pid


def meta_from_manifest(eid: str, manifest: Dict[str, dict]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    payload = manifest.get(eid, {}) or {}
    store_id = payload.get("store_id") or payload.get("store") or None
    day_id = payload.get("day_id") or payload.get("day") or None
    camera = payload.get("camera") or None
    ts = payload.get("timestamp") or payload.get("ts") or None
    images = payload.get("images") or []
    image = images[0] if images else None
    return store_id, day_id, camera, ts, image, payload.get("direction")


def build_pair_record(
    i: int,
    j: int,
    sim: float,
    ids: List[str],
    pid_of: List[Optional[str]],
    manifest: Dict[str, dict],
    risk_type: str,
) -> PairRecord:
    eid_a = ids[i]
    eid_b = ids[j]
    pid_a = pid_of[i]
    pid_b = pid_of[j]
    store_a, day_a, cam_a, ts_a, img_a, _ = meta_from_manifest(eid_a, manifest)
    store_b, day_b, cam_b, ts_b, img_b, _ = meta_from_manifest(eid_b, manifest)
    return PairRecord(
        entry_id_a=eid_a,
        entry_id_b=eid_b,
        pid_a=str(pid_a),
        pid_b=str(pid_b),
        same_pid=int(pid_a == pid_b),
        sim=float(sim),
        risk_type=risk_type,
        store_a=store_a,
        store_b=store_b,
        day_a=day_a,
        day_b=day_b,
        camera_a=cam_a,
        camera_b=cam_b,
        ts_a=ts_a,
        ts_b=ts_b,
        image_a=img_a,
        image_b=img_b,
    )


def find_hard_positives(
    V: np.ndarray,
    ids: List[str],
    pid_of: List[Optional[str]],
    idxs_by_pid: Dict[str, List[int]],
    manifest: Dict[str, dict],
    pos_threshold: float,
) -> Dict[Tuple[int, int], PairRecord]:
    """
    Same person_id but similarity below pos_threshold.
    """
    pairs: Dict[Tuple[int, int], PairRecord] = {}
    num_pids = len(idxs_by_pid)
    print(f"[pos] starting search for hard positives across {num_pids} person clusters...")
    for idx, (pid, idxs) in enumerate(idxs_by_pid.items()):
        if (idx + 1) % 500 == 0:
            print(f"[pos] ... processed {idx + 1}/{num_pids} person clusters")
        if len(idxs) < 2:
            continue
        sub = V[idxs]  # (m,D)
        S_sub = sub @ sub.T
        m = len(idxs)
        for a_local in range(m):
            for b_local in range(a_local + 1, m):
                i = idxs[a_local]
                j = idxs[b_local]
                sim = float(S_sub[a_local, b_local])
                if sim > pos_threshold:
                    continue
                key = (i, j) if i < j else (j, i)
                rec = build_pair_record(i, j, sim, ids, pid_of, manifest, risk_type="hard_pos")
                existing = pairs.get(key)
                # keep the *hardest* version (lowest sim)
                if existing is None or sim < existing.sim:
                    pairs[key] = rec
    print(f"[pos] found {len(pairs)} candidate pairs.")
    return pairs


def find_hard_negatives(
    V: np.ndarray,
    ids: List[str],
    pid_of: List[Optional[str]],
    manifest: Dict[str, dict],
    neg_threshold: float,
    topk_neg: int,
) -> Dict[Tuple[int, int], PairRecord]:
    """
    Different person_id but similarity above neg_threshold.

    To keep things tractable, we only consider the top-K nearest neighbors per anchor.
    """
    N, _ = V.shape
    pairs: Dict[Tuple[int, int], PairRecord] = {}

    print(f"[neg] starting search for hard negatives across {N} entries...")
    for i in range(N):
        if (i + 1) % 500 == 0:
            print(f"[neg] ... processed {i + 1}/{N} anchors")
        pid_i = pid_of[i]
        if pid_i is None:
            continue

        sims = V[i] @ V.T  # (N,)
        sims[i] = -np.inf  # ignore self

        # sort neighbors by similarity (descending)
        order = np.argsort(-sims)
        # scan until we've considered at least topk_neg negatives or hit a low-sim neighbor
        neg_seen = 0
        for j in order:
            if j == i:
                continue
            pid_j = pid_of[j]
            if pid_j is None:
                continue
            if pid_j == pid_i:
                # same pid → genuine; skip for negative mining
                continue
            sim = float(sims[j])
            if sim < neg_threshold:
                # since order is descending, we can break early
                break
            key = (i, j) if i < j else (j, i)
            rec = build_pair_record(i, j, sim, ids, pid_of, manifest, risk_type="hard_neg")
            existing = pairs.get(key)
            if existing is None or sim > existing.sim:
                pairs[key] = rec
            neg_seen += 1
            if neg_seen >= topk_neg:
                break
    print(f"[neg] found {len(pairs)} candidate pairs.")
    return pairs


def mark_alias_candidates(pairs: Dict[Tuple[int, int], PairRecord], strong_neg_threshold: float = 0.93) -> None:
    """
    Upgrade some hard_neg pairs to 'alias_candidate' if they look like
    same physical person across days:

      - same store_id
      - different day_id
      - high similarity (>= strong_neg_threshold)
    """
    for rec in pairs.values():
        if rec.risk_type != "hard_neg":
            continue
        if rec.store_a and rec.store_b and rec.store_a == rec.store_b:
            if rec.day_a and rec.day_b and rec.day_a != rec.day_b:
                if rec.sim >= strong_neg_threshold:
                    rec.risk_type = "alias_candidate"


def write_jsonl(pairs: Iterable[PairRecord], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec.to_row(), ensure_ascii=False) + "\n")


def write_csv(pairs: Iterable[PairRecord], out_path: Path) -> None:
    pairs = list(pairs)
    if not pairs:
        out_path.write_text("")
        return
    fieldnames = list(pairs[0].to_row().keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in pairs:
            w.writerow(rec.to_row())


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract high-risk GT pairs (hard pos/neg) from entry embeddings.")
    ap.add_argument("--vecs", type=Path, required=True, help="entry_vectors.npy from infer_embed.py")
    ap.add_argument("--ids", type=Path, required=True, help="entry_ids.txt from infer_embed.py")
    ap.add_argument("--manifest", type=Path, required=True, help="manifest.json used to train (entry_id -> payload)")
    ap.add_argument("--gt", type=Path, required=True, help="multi_gt.json mapping entry_id -> person_id")
    ap.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to write output files.")
    ap.add_argument("--neg-threshold", type=float, default=0.88, help="similarity threshold for hard negatives")
    ap.add_argument("--pos-threshold", type=float, default=0.75, help="similarity upper bound for hard positives")
    ap.add_argument("--topk-neg", type=int, default=20, help="max negative neighbors per anchor to inspect")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "pairs_hard.jsonl"
    out_csv = args.output_dir / "pairs_hard.csv"

    V, ids = load_embeddings(args.vecs, args.ids)
    manifest = load_manifest(args.manifest)
    entry_to_pid = load_gt(args.gt)
    pid_of, idxs_by_pid = build_pid_index(ids, entry_to_pid)

    print(f"[data] {len(ids)} entries, dim={V.shape[1]}")
    print(f"[cfg] neg_threshold={args.neg_threshold:.3f} pos_threshold={args.pos_threshold:.3f} topk_neg={args.topk_neg}")

    hard_pos = find_hard_positives(V, ids, pid_of, idxs_by_pid, manifest, pos_threshold=args.pos_threshold)
    hard_neg = find_hard_negatives(V, ids, pid_of, manifest, neg_threshold=args.neg_threshold, topk_neg=args.topk_neg)

    # merge
    all_pairs: Dict[Tuple[int, int], PairRecord] = {}
    all_pairs.update(hard_pos)
    # if a pair is both (possible only via weird GT), prefer hard_pos record
    for key, rec in hard_neg.items():
        all_pairs.setdefault(key, rec)

    # mark alias candidates (strong hard negatives across days)
    mark_alias_candidates(all_pairs)

    # sort for convenience: hard_neg/alias by descending sim, hard_pos by ascending sim
    neg_like = [rec for rec in all_pairs.values() if rec.risk_type in ("hard_neg", "alias_candidate")]
    pos_like = [rec for rec in all_pairs.values() if rec.risk_type == "hard_pos"]

    neg_like.sort(key=lambda r: -r.sim)
    pos_like.sort(key=lambda r: r.sim)

    ordered = neg_like + pos_like

    print(f"[done] found {len(neg_like)} hard neg / alias candidates and {len(pos_like)} hard positives.")
    print(f"        writing {out_jsonl} and {out_csv}")

    write_jsonl(ordered, out_jsonl)
    write_csv(ordered, out_csv)


if __name__ == "__main__":
    main()
