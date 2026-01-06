# data_utils.py
from typing import Dict, Any, List
import os

os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")

import numpy as np
import json
from pathlib import Path
from frame_filter import l2norm_np

def load_attr_schema(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def vec_from_schema(attrs: Dict[str, Any], schema: Dict[str, Any]) -> np.ndarray:
    if not schema:
        return np.zeros((0,), dtype=np.float32)
    vec: List[np.ndarray] = []
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
            s = float(arr.sum()); arr = arr/(s+1e-12) if s>0 else arr
            vec.append(arr)
        elif ftype == "float":
            lo, hi = f.get("range",[0.0,1.0])
            if v is None: val = 0.0
            else:
                try: val = (float(v)-float(lo))/max(1e-6, float(hi)-float(lo))
                except: val = 0.0
                val = max(0.0, min(1.0, val))
            vec.append(np.array([val], dtype=np.float32))
        elif ftype == "bool":
            val = 0.0 if v is None else (1.0 if bool(v) else 0.0)
            vec.append(np.array([val], dtype=np.float32))
        else:
            vec.append(np.array([0.0], dtype=np.float32))
    return np.concatenate(vec, axis=0).astype(np.float32) if vec else np.zeros((0,), dtype=np.float32)

