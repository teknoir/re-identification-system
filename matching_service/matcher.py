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
from .metric_model import load_entry_encoder, encode_numpy
from .data_utils import load_attr_schema, vec_from_schema

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

class ReEntryMatcher:
    def __init__(
        self,
        model_ckpt_path: str,
        attr_schema_path: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        mongo_db: str = "reid_service",
        entries_collection: str = "entries",
        events_collection: str = "line-crossings",
        threshold: float = 0.88,
        topk: int = 20,
    ):
        # load model (new encoder2 checkpoint format)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ck = load_entry_encoder(model_ckpt_path, device=self.device)
        self.model.eval()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        self.dim = int(self.ck["emb_dim"])

        self.attr_schema = load_attr_schema(attr_schema_path) if attr_schema_path else None

        self.mongo = None
        self.db = None
        if mongo_uri:
            self.mongo = MongoClient(mongo_uri)
            self.db = self.mongo[mongo_db]

        self.threshold = float(os.getenv("THRESHOLD", threshold))
        self.topk = int(os.getenv("TOPK", topk))

        self.by_day_store: Dict[str, Dict[str, DayIndex]] = {}
        self.entries_collection = entries_collection
        self.events_collection = events_collection

    @staticmethod
    def _store_from_entry(entry_id: str) -> str:
        return entry_id.split("-", 1)[0] if entry_id else "unknown-store"

    def ensure_day_store(self, day_id: str, store_id: str):
        day = self.by_day_store.setdefault(day_id, {})
        if store_id not in day:
            day[store_id] = DayIndex(self.dim)
        return day[store_id]

    def encode_entry(self, embeddings: List[List[float]], attrs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        vis_embeds = np.stack([np.asarray(v, np.float32).reshape(-1) for v in embeddings], axis=0)
        # ensure per-frame vectors are normalized to stay consistent with training
        norms = np.linalg.norm(vis_embeds, axis=1, keepdims=True) + 1e-10
        vis_embeds = vis_embeds / norms
        mask = np.ones((vis_embeds.shape[0],), dtype=bool)

        attr_vec = vec_from_schema(attrs or {}, self.attr_schema) if self.attr_schema else np.zeros((0,), np.float32)
        attr_dim = int(self.ck["attr_dim"])
        if attr_vec.size == 0 and attr_dim > 0:
            attr_vec = np.zeros((attr_dim,), dtype=np.float32)
        if attr_vec.shape[0] != attr_dim:
            raise ValueError(f"attr vector dim {attr_vec.shape[0]} != expected {attr_dim}")

        z = encode_numpy(self.model, vis_embeds, mask, attr_vec, device=self.device)
        return z.astype(np.float32, copy=False)

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
        idx = self.ensure_day_store(day_id, resolved_store)

        z = self.encode_entry(embeddings, attrs)

        # query before add (avoid matching ourselves)
        decision = {"status": "new", "match_id": None, "score": None, "score2": None}
        if idx.index.ntotal > 0:
            K = topk or self.topk
            sims, nn_idx, ids = idx.search(z, K)
            nn1 = float(sims[0]); nn2 = float(sims[1]) if len(sims) > 1 else -1.0
            id1 = ids[int(nn_idx[0])]
            is_match = nn1 >= self.threshold
            if is_match:
                decision.update({"status": "match", "match_id": id1})
            decision.update({"score": nn1, "score2": nn2})

        # add to FAISS
        idx.add(z, entry_id)

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
                "files": image_uris or [],
                "images": image_uris or [],
                "vis": {"per_image_dim": len(embeddings[0]) if embeddings else 0, "embeddings": embeddings},
                "embeddings": embeddings or [],
                "attrs": attrs or {},
            }
            self.db[self.entries_collection].replace_one({"_id": entry_id}, doc, upsert=True)

        return decision

    def rebuild_from_mongo(self, day_id: str) -> Dict[str, Any]:
        if self.db is None:
            return {"ok": False, "error": "Mongo not configured"}
        self.by_day_store[day_id] = {}
        cur = self.db[self.entries_collection].find({"day_id": day_id}, {"_id": 1, "store_id": 1, "vis.embeddings": 1, "attrs": 1, "images": 1})
        cnt = 0
        for doc in cur:
            eid = doc["_id"]
            store = doc.get("store_id") or self._store_from_entry(eid)
            idx = self.ensure_day_store(day_id, store)
            embs = doc.get("vis", {}).get("embeddings") or []
            attrs = doc.get("attrs") or {}
            if not embs:
                continue
            z = self.encode_entry(embs, attrs)
            idx.add(z, eid)
            cnt += 1
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
                is_match = nn1 >= self.threshold
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
