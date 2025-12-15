# data_loaders.py
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

def vec_from_schema(attrs: Dict[str, Any], schema: Dict[str, Any]) -> np.ndarray:
    if not schema:
        return np.zeros((0,), dtype=np.float32)
    vec = []
    for f in schema["fields"]:
        name = f["name"]; ftype = f["type"]
        v = (attrs or {}).get(name, None)
        if ftype in ("enum","multilabel"):
            vocab = f.get("values", [])
            if ftype == "enum":
                one = np.zeros((len(vocab),), dtype=np.float32)
                if v in vocab:
                    one[vocab.index(v)] = 1.0
                vec.append(one)
            else:
                one = np.zeros((len(vocab),), dtype=np.float32)
                if isinstance(v, list):
                    for it in v:
                        if it in vocab:
                            one[vocab.index(it)] = 1.0
                vec.append(one)
        elif ftype == "prob_dict":
            vocab = f.get("values", [])
            arr = np.array([float((v or {}).get(k, 0.0)) for k in vocab], dtype=np.float32)
            s = float(arr.sum())
            arr = arr/(s+1e-12) if s>0 else arr
            vec.append(arr)
        elif ftype == "float":
            lo, hi = f.get("range",[0.0,1.0])
            if v is None: val = 0.0
            else:
                try:
                    val = (float(v)-float(lo))/max(1e-6, float(hi)-float(lo))
                except:
                    val = 0.0
                val = max(0.0, min(1.0, val))
            vec.append(np.array([val], dtype=np.float32))
        elif ftype == "bool":
            val = 0.0 if v is None else (1.0 if bool(v) else 0.0)
            vec.append(np.array([val], dtype=np.float32))
        else:
            vec.append(np.array([0.0], dtype=np.float32))
    return np.concatenate(vec, axis=0).astype(np.float32) if vec else np.zeros((0,), dtype=np.float32)

def l2norm(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x) + 1e-12)
    return x / n

class EntryDataset(Dataset):
    """Loads entries from manifest + GT labels for metric learning."""
    def __init__(self, manifest_path: Path, gt_path: Path, schema_path: Optional[Path]=None):
        self.manifest = json.loads(Path(manifest_path).read_text())
        gt = json.loads(Path(gt_path).read_text())

        # entry_id -> person_id
        # The ground truth file is a direct mapping from entry_id to person_id.
        self.entry_to_pid: Dict[str, str] = gt

        self.schema = json.loads(Path(schema_path).read_text()) if schema_path else None

        self.items: List[Tuple[str, np.ndarray, np.ndarray, int]] = []
        pid_vocab = {pid:i for i, pid in enumerate(sorted(set(self.entry_to_pid.values())))}

        for eid, payload in self.manifest.items():
            if eid not in self.entry_to_pid:
                continue
            emb_list = payload.get("embeddings") or []
            if not emb_list:
                continue
            vis = np.stack([l2norm(np.asarray(v, dtype=np.float32).reshape(-1)) for v in emb_list], axis=0)  # (N, Dv)
            attrs = payload.get("attrs") or {}
            avec = vec_from_schema(attrs, self.schema) if self.schema else np.zeros((0,), dtype=np.float32)
            y = pid_vocab[self.entry_to_pid[eid]]
            self.items.append((eid, vis, avec, y))

        self.num_classes = len(pid_vocab)
        # reverse index: class -> list of dataset indices
        self.pid_to_indices: Dict[int, List[int]] = {}
        for i, (_, _, _, y) in enumerate(self.items):
            self.pid_to_indices.setdefault(int(y), []).append(i)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        eid, vis, avec, y = self.items[idx]
        vis_t = torch.from_numpy(vis)                # (N, Dv)
        attr_t = torch.from_numpy(avec) if avec.size else torch.empty(0)
        y_t = torch.tensor(y, dtype=torch.long)
        return {"entry_id": eid, "vis": vis_t, "attr": attr_t, "y": y_t}

def collate_variable_sets(batch):
    # pad sets to same N and build masks for valid frames
    maxN = max(b["vis"].shape[0] for b in batch)
    Dv = batch[0]["vis"].shape[1]
    vis_batch, mask_batch = [], []
    for b in batch:
        v = b["vis"]
        n = v.shape[0]
        if n < maxN:
            pad = torch.zeros((maxN - n, Dv), dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        m = torch.zeros((maxN,), dtype=torch.bool); m[:n] = True
        vis_batch.append(v.unsqueeze(0))
        mask_batch.append(m.unsqueeze(0))
    vis_batch = torch.cat(vis_batch, dim=0)   # (B, maxN, Dv)
    mask_batch = torch.cat(mask_batch, dim=0) # (B, maxN)

    attr_list = [b["attr"].unsqueeze(0) if b["attr"].numel()>0 else torch.zeros((1,0)) for b in batch]
    attr_batch = torch.cat(attr_list, dim=0)
    labels = torch.stack([b["y"] for b in batch], dim=0)
    entry_ids = [b["entry_id"] for b in batch]
    return {"entry_ids": entry_ids, "vis": vis_batch, "mask": mask_batch, "attr": attr_batch, "y": labels}

class PKSampler(Sampler[int]):
    """
    Ensures each 'batch' is formed of P identities × K samples (so effective batch_size = P*K).
    DataLoader should be given batch_size=P*K and this sampler.
    """
    def __init__(self, dataset: EntryDataset, P: int = 32, K: int = 2, seed: int = 13):
        self.ds = dataset
        self.P = max(1, int(P))
        self.K = max(2, int(K))
        self.rng = random.Random(seed)
        self.pids = list(dataset.pid_to_indices.keys())

    def __iter__(self):
        pids = self.pids[:]
        self.rng.shuffle(pids)
        # number of groups
        if not pids:
            return
        # round up groups
        G = (len(pids) + self.P - 1) // self.P
        for g in range(G):
            chunk = pids[g*self.P:(g+1)*self.P]
            if len(chunk) < self.P:
                chunk += pids[:(self.P - len(chunk))]
            batch_indices = []
            for pid in chunk:
                pool = self.ds.pid_to_indices.get(pid, [])
                if not pool:
                    continue
                if len(pool) < self.K:
                    select = [self.rng.choice(pool) for _ in range(self.K)]
                else:
                    select = self.rng.sample(pool, self.K)
                batch_indices.extend(select)
            self.rng.shuffle(batch_indices)
            for idx in batch_indices:
                yield idx

    def __len__(self):
        if not self.pids:
            return 0
        G = (len(self.pids) + self.P - 1) // self.P
        return G * (self.P * self.K)
