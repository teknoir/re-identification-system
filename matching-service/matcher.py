# matcher.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import os
from datetime import datetime

os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))

import torch
import faiss
import numpy as np
from pymongo import MongoClient
from .metric_model import EntryEncoder
from .data_utils import load_attr_schema, vec_from_schema, l2norm_np
import json
from pathlib import Path
import logging

class DayIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine if inputs L2-normalized
        self.ids: List[str] = []

    def add(self, vec: np.ndarray, entry_id: str):
        self.index.add(vec.reshape(1, -1).astype(np.float32))
        self.ids.append(entry_id)

    def search(self, vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        D, I = self.index.search(vec.reshape(1, -1).astype(np.float32), topk)
        return D[0], I[0], self.ids
    
    def save(self, index_path: Path, data_path: Path, entry_count: int):
        faiss.write_index(self.index, str(index_path))
        with open(data_path, "w") as f:
            json.dump({"ids": self.ids, "entry_count": entry_count}, f)

    @classmethod
    def load(cls, index_path: Path, data_path: Path, dim: int) -> "DayIndex":
        day_index = cls(dim)
        day_index.index = faiss.read_index(str(index_path))
        with open(data_path, "r") as f:
            data = json.load(f)
            day_index.ids = data["ids"]
        return day_index

class ReEntryMatcher:
    def __init__(
        self,
        model_ckpt_path: str,
        attr_schema_path: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        mongo_db: str = "retail_reid",
        events_collection: str = "line-crossings",
        entries_collection: str = "observations",
        threshold: float = 0.88,
        margin: float = 0.02,
        topk: int = 20,
    ):
        # load model
        ck = torch.load(model_ckpt_path, map_location="cpu")
        self.model = EntryEncoder(ck["vis_dim"], ck["attr_dim"], ck["emb_dim"], ck["use_attention"], ck["dropout"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        self.dim = ck["emb_dim"]

        self.attr_schema = load_attr_schema(attr_schema_path) if attr_schema_path else None

        self.mongo = None
        self.db = None
        if mongo_uri:
            self.mongo = MongoClient(mongo_uri)
            self.db = self.mongo[mongo_db]

        self.threshold = float(os.getenv("THRESHOLD", threshold))
        self.margin = float(os.getenv("MARGIN", margin))
        self.topk = int(os.getenv("TOPK", topk))

        self.by_day_store: Dict[str, Dict[str, DayIndex]] = {}
        self.events_collection = events_collection
        self.entries_collection = entries_collection

        self.cache_dir = Path("matching-service/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_entry_count(self, day_id: str, store_id: str) -> int:
        if self.db is None:
            return 0
        return self.db[self.entries_collection].count_documents({"day_id": day_id, "store_id": store_id})

    def save_to_cache(self, day_id: str, store_id: str):
        day_store = self.by_day_store.get(day_id, {})
        day_index = day_store.get(store_id)
        if not day_index:
            return

        entry_count = self.get_entry_count(day_id, store_id)
        
        index_path = self.cache_dir / f"{day_id}_{store_id}.index"
        data_path = self.cache_dir / f"{day_id}_{store_id}.json"
        
        day_index.save(index_path, data_path, entry_count)

    def load_from_cache(self, day_id: str, store_id: str) -> bool:
        index_path = self.cache_dir / f"{day_id}_{store_id}.index"
        data_path = self.cache_dir / f"{day_id}_{store_id}.json"

        if not index_path.exists() or not data_path.exists():
            return False

        with open(data_path, "r") as f:
            data = json.load(f)

        # Cache invalidation check
        entry_count = data.get("entry_count", -1)
        if self.get_entry_count(day_id, store_id) != entry_count:
            logging.info(f"Cache invalid for day {day_id}, store {store_id}. Rebuilding.")
            return False
            
        day_index = DayIndex.load(index_path, data_path, self.dim)
        
        day = self.by_day_store.setdefault(day_id, {})
        day[store_id] = day_index
        
        logging.info(f"Loaded index for day {day_id}, store {store_id} from cache.")
        return True

    @staticmethod
    def _store_from_entry(entry_id: str) -> str:
        return entry_id.split("-", 1)[0] if entry_id else "unknown-store"

    def ensure_day_store(self, day_id: str, store_id: str):
        day = self.by_day_store.setdefault(day_id, {})
        if store_id not in day:
            day[store_id] = DayIndex(self.dim)
        return day[store_id]

    def encode_entry(self, embeddings: List[List[float]], attrs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        V = np.stack([l2norm_np(np.asarray(v, np.float32).reshape(-1)) for v in embeddings], axis=0)  # (N,Dv)
        avec = vec_from_schema(attrs or {}, self.attr_schema) if self.attr_schema else np.zeros((0,), np.float32)

        with torch.no_grad():
            vis_t = torch.from_numpy(V).unsqueeze(0).float()
            mask = torch.ones((1, V.shape[0]), dtype=torch.bool)
            attr_t = torch.from_numpy(avec).unsqueeze(0).float() if avec.size>0 else None
            z = self.model(vis_t, attr_t, mask=mask).squeeze(0).cpu().numpy()
        return l2norm_np(z.astype(np.float32))

    def query_then_add(
        self,
        day_id: str,
        entry_id: str,
        store_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        direction: Optional[str] = None,
        camera: Optional[str] = None,
        image_uris: Optional[List[str]] = None,
        embeddings: List[List[float]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        topk: Optional[int] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        resolved_store = store_id or self._store_from_entry(entry_id)

        # Check if index is in memory, if not, try to load from cache or rebuild
        if self.by_day_store.get(day_id, {}).get(resolved_store):
            logging.info(f"Using in-memory index for day {day_id}, store {resolved_store}.")
        elif not self.load_from_cache(day_id, resolved_store):
            self.rebuild_from_mongo(day_id)

        idx = self.ensure_day_store(day_id, resolved_store)

        z = self.encode_entry(embeddings, attrs)

        # query before add (avoid matching ourselves)
        decision = {"status": "new", "match_id": None, "score": None, "score2": None}
        if idx.index.ntotal > 0:
            K = topk or self.topk
            sims, nn_idx, ids = idx.search(z, K)
            nn1 = float(sims[0]); nn2 = float(sims[1]) if len(sims) > 1 else -1.0
            id1 = ids[int(nn_idx[0])]
            is_match = (nn1 >= self.threshold) and ((nn1 - nn2) >= self.margin)
            if is_match:
                decision.update({"status": "match", "match_id": id1})
            decision.update({"score": nn1, "score2": nn2})

        # add to FAISS
        idx.add(z, entry_id)

        # save to cache after adding
        self.save_to_cache(day_id, resolved_store)

        # optional persistence (raw inputs)
        if persist and self.db is not None:
            doc = {
                "_id": entry_id,
                "day_id": day_id,
                "store_id": resolved_store,
                "alert_id": alert_id,
                "timestamp": timestamp,
                "direction": direction,
                "camera": camera,
                "images": image_uris or [],
                "vis": {"per_image_dim": len(embeddings[0]) if embeddings else 0, "embeddings": embeddings},
                "attrs": attrs or {},
            }
            self.db[self.entries_collection].replace_one({"_id": entry_id}, doc, upsert=True)

        return decision

    def rebuild_from_mongo(self, day_id: str) -> Dict[str, Any]:
        logging.info(f"Rebuilding index for day {day_id} from Mongo.")
        if self.db is None:
            return {"ok": False, "error": "Mongo not configured"}
        self.by_day_store[day_id] = {}
        cur = self.db[self.entries_collection].find({"day_id": day_id}, {"_id": 1, "store_id": 1, "vis.embeddings": 1, "attrs": 1, "images": 1})
        cnt = 0
        stores_rebuilt = set()
        for doc in cur:
            eid = doc["_id"]
            store = doc.get("store_id") or self._store_from_entry(eid)
            stores_rebuilt.add(store)
            idx = self.ensure_day_store(day_id, store)
            embs = doc.get("vis", {}).get("embeddings") or []
            attrs = doc.get("attrs") or {}
            if not embs:
                continue
            z = self.encode_entry(embs, attrs)
            idx.add(z, eid)
            cnt += 1
        
        for store_id in stores_rebuilt:
            self.save_to_cache(day_id, store_id)
            
        return {"ok": True, "count": cnt}

    @staticmethod
    def _ts_key(value: Optional[str]) -> datetime:
        if not value:
            return datetime.min
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    def _load_event_metadata(self, entry_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not entry_ids or self.db is None or not self.events_collection:
            return out
        cur = self.db[self.events_collection].find(
            {"metadata.id": {"$in": entry_ids}},
            {"metadata.timestamp": 1, "metadata.id": 1, "metadata.annotations": 1, "data.peripheral.name": 1, "data.burst": 1, "data.filename": 1},
        )
        for doc in cur:
            meta = doc.get("metadata") or {}
            eid = meta.get("id")
            if not eid:
                continue
            annotations = meta.get("annotations") or {}
            direction = annotations.get("teknoir.org/linedir") or annotations.get("linedir")
            images = doc.get("data", {}).get("burst") or []
            if not images and doc.get("data", {}).get("filename"):
                images = [doc["data"]["filename"]]
            out[eid] = {
                "timestamp": meta.get("timestamp"),
                "direction": direction,
                "camera": (doc.get("data") or {}).get("peripheral", {}).get("name"),
                "images": images,
            }
        return out

    def build_manifest(self, day_id: str, store_id: str, entry_id: Optional[str] = None, cameras: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.db is None:
            raise ValueError("Mongo not configured")

        # Check if index is in memory, if not, try to load from cache or rebuild
        if self.by_day_store.get(day_id, {}).get(store_id):
            logging.info(f"Using in-memory index for day {day_id}, store {store_id}.")
        elif not self.load_from_cache(day_id, store_id):
            self.rebuild_from_mongo(day_id)

        query = {"day_id": day_id, "store_id": store_id}
        if cameras:
            query["camera"] = {"$in": cameras}
        docs = list(self.db[self.entries_collection].find(query))
        if not docs:
            return {"day_id": day_id, "store_id": store_id, "person_count": 0, "event_count": 0, "people": []}
        entry_ids = [doc["_id"] for doc in docs]
        meta_map = self._load_event_metadata(entry_ids)
        records = []
        for doc in docs:
            embs = (doc.get("vis") or {}).get("embeddings") or []
            if not embs:
                continue
            eid = doc["_id"]
            meta = meta_map.get(eid, {})
            timestamp = meta.get("timestamp") or doc.get("timestamp")
            direction = meta.get("direction") or doc.get("direction")
            camera = meta.get("camera") or doc.get("camera")
            images = doc.get("images") or meta.get("images") or []
            records.append(
                {
                    "entry_id": eid,
                    "embeddings": embs,
                    "attrs": doc.get("attrs") or {},
                    "alert_id": doc.get("alert_id"),
                    "images": images,
                    "timestamp": timestamp,
                    "direction": direction,
                    "camera": camera,
                }
            )
        if not records:
            return {"day_id": day_id, "store_id": store_id, "person_count": 0, "event_count": 0, "people": []}
        records.sort(key=lambda r: (self._ts_key(r["timestamp"]), r["entry_id"]))

        temp_index = faiss.IndexFlatIP(self.dim)
        assignments: List[int] = []
        people: List[Dict[str, Any]] = []

        for rec in records:
            vec = self.encode_entry(rec["embeddings"], rec["attrs"])
            cluster_idx = None
            nn1 = None
            nn2 = None
            if temp_index.ntotal > 0:
                K = min(temp_index.ntotal, max(2, self.topk))
                sims, idxs = temp_index.search(vec.reshape(1, -1).astype(np.float32), K)
                nn1 = float(sims[0][0])
                nn2 = float(sims[0][1]) if K > 1 else -1.0
                is_match = (nn1 >= self.threshold) and ((nn1 - nn2) >= self.margin)
                if is_match:
                    cluster_idx = assignments[int(idxs[0][0])]
            if cluster_idx is None:
                cluster_idx = len(people)
                first_seen = rec["timestamp"]
                people.append(
                    {"person_id": f"{store_id}-{cluster_idx+1:04d}", "first_seen": first_seen, "events": []}
                )
            people[cluster_idx]["events"].append(
                {
                    "entry_id": rec["entry_id"],
                    "timestamp": rec["timestamp"],
                    "direction": rec["direction"],
                    "camera": rec["camera"],
                    "alert_id": rec["alert_id"],
                    "images": rec["images"],
                    "score": nn1,
                    "score2": nn2,
                    "attrs": rec["attrs"],
                    "embeddings": rec["embeddings"],
                }
            )
            if rec["timestamp"]:
                if (
                    people[cluster_idx]["first_seen"] is None
                    or self._ts_key(rec["timestamp"]) < self._ts_key(people[cluster_idx]["first_seen"])
                ):
                    people[cluster_idx]["first_seen"] = rec["timestamp"]
            temp_index.add(vec.reshape(1, -1).astype(np.float32))
            assignments.append(cluster_idx)

        if entry_id:
            people = [p for p in people if any(ev["entry_id"] == entry_id for ev in p["events"])]

        for person in people:
            person["events"].sort(key=lambda ev: (self._ts_key(ev["timestamp"]), ev["entry_id"]))

        return {
            "day_id": day_id,
            "store_id": store_id,
            "person_count": len(people),
            "event_count": len(records),
            "people": people,
        }
    def fetch_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        if self.db is None:
            return None
        return self.db[self.entries_collection].find_one({"_id": entry_id})

    def _doc_to_event(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entry_id": doc.get("_id"),
            "day_id": doc.get("day_id"),
            "store_id": doc.get("store_id"),
            "alert_id": doc.get("alert_id"),
            "timestamp": doc.get("timestamp"),
            "direction": doc.get("direction"),
            "camera": doc.get("camera"),
            "images": doc.get("images") or [],
            "attrs": doc.get("attrs") or {},
            "embeddings": (doc.get("vis") or {}).get("embeddings") or [],
        }

    def compare_entries(self, entry_a: str, entry_b: str) -> Dict[str, Any]:
        doc_a = self.fetch_entry(entry_a)
        doc_b = self.fetch_entry(entry_b)
        if doc_a is None or doc_b is None:
            missing = entry_a if doc_a is None else entry_b
            raise ValueError(f"entry {missing} not found in persistence")
        embs_a = (doc_a.get("vis") or {}).get("embeddings") or []
        embs_b = (doc_b.get("vis") or {}).get("embeddings") or []
        if not embs_a or not embs_b:
            raise ValueError("missing embeddings for comparison")
        attrs_a = doc_a.get("attrs") or {}
        attrs_b = doc_b.get("attrs") or {}
        vec_a = self.encode_entry(embs_a, attrs_a)
        vec_b = self.encode_entry(embs_b, attrs_b)
        score = float(np.dot(vec_a, vec_b))
        return {
            "score": score,
            "entry_a": self._doc_to_event(doc_a),
            "entry_b": self._doc_to_event(doc_b),
        }
