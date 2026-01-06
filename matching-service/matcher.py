# matcher.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List, Union
import os
from datetime import datetime, timedelta

os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("LOG_MATCH_DECISIONS", "1")
fusion_mode ="xattn"


import torch
import faiss
import numpy as np
from pymongo import MongoClient
from metric_model import EntryEncoder, CrossAttentionEntryEncoder
from data_utils import load_attr_schema, vec_from_schema
from frame_filter import filter_frames, l2norm_np
import json
from pathlib import Path
import logging
import pytz

def _get_day_id_for_utc_timestamp(timestamp: str, timezone_str: str = "America/New_York") -> str:
    """Converts a UTC timestamp string to a local date string (YYYY-MM-DD)."""
    try:
        utc_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        # Fallback for non-iso timestamps or other errors
        return datetime.utcnow().strftime("%Y-%m-%d")
    
    local_tz = pytz.timezone(timezone_str)
    local_dt = utc_dt.astimezone(local_tz)
    return local_dt.strftime("%Y-%m-%d")


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
        threshold: float,
        margin: float,
        topk: int,
        empty_after_filter_policy: str = "skip",
        attr_schema_path: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        mongo_db: str = "retail_reid",
        events_collection: str = "line-crossings",
        entries_collection: str = "observations",
    ):
        # load model
        ck = torch.load(model_ckpt_path, map_location="cpu")
        # self.model = EntryEncoder(ck["vis_dim"], ck["attr_dim"], ck["emb_dim"], ck["use_attention"], ck["dropout"])

        if fusion_mode == "xattn":
            self.model = CrossAttentionEntryEncoder(
                ck["vis_dim"],
                ck["attr_dim"],
                ck["emb_dim"],
                ck["use_attention"],
                ck["dropout"],
            )
        else:
            self.model = EntryEncoder(
                ck["vis_dim"],
                ck["attr_dim"],
                ck["emb_dim"],
                ck["use_attention"],
                ck["dropout"],
            )

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

        # Use passed-in parameters directly
        self.threshold = threshold
        self.margin = margin
        self.topk = topk

        # How to handle entries where frame filtering drops everything
        # - 'skip': do not match and do not index (training-aligned; safest for FP control)
        # - 'fallback': use unfiltered frames (legacy behavior; can increase FPs)
        self.empty_after_filter_policy = (empty_after_filter_policy or "skip").strip().lower()
        if self.empty_after_filter_policy not in ("skip", "fallback"):
            logging.warning(
                "Unknown empty_after_filter_policy=%r; defaulting to 'skip'",
                self.empty_after_filter_policy,
            )
            self.empty_after_filter_policy = "skip"

        # Metrics
        self.total_processed = 0
        self.total_empty_after_filter = 0

        self.by_day_store: Dict[str, Dict[str, DayIndex]] = {}
        self.events_collection = events_collection
        self.entries_collection = entries_collection

        self.cache_dir = Path("matching-service/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime metrics."""
        perc = 0.0
        if self.total_processed > 0:
            perc = (self.total_empty_after_filter / self.total_processed) * 100.0
        return {
            "total_processed": self.total_processed,
            "total_empty_after_filter": self.total_empty_after_filter,
            "empty_after_filter_percent": round(perc, 2),
        }

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

    def encode_entry(
        self,
        embeddings: List[List[float]],
        attrs: Optional[Dict[str, Any]] = None,
        *,
        return_info: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Encode a single entry into the fused embedding space.

        Note on training/runtime parity:
          - Training skips entries that become empty after frame filtering.
          - Runtime historically fell back to unfiltered frames (which can create FPs).
        This function supports both via `self.empty_after_filter_policy`.
        """
        info: Dict[str, Any] = {
            "orig_frames": 0,
            "filtered_frames": 0,
            "frames_used": 0,
            "kept_indices": [],
            "empty_after_filter": False,
            "empty_policy": self.empty_after_filter_policy,
            "used_unfiltered_fallback": False,
            "skip_match_and_index": False,
        }

        if not embeddings:
            z0 = np.zeros((self.dim,), dtype=np.float32)
            info.update(
                {
                    "empty_after_filter": True,
                    "skip_match_and_index": True,
                    "reason": "no_embeddings",
                }
            )
            return (z0, info) if return_info else z0

        # 1) Build (N, D) numpy array
        vis_np = np.asarray(embeddings, dtype=np.float32)
        if vis_np.ndim == 1:
            vis_np = vis_np.reshape(1, -1)
        else:
            vis_np = vis_np.reshape(vis_np.shape[0], -1)

        info["orig_frames"] = int(vis_np.shape[0])

        # 2) (Legacy) L2-normalize. Note: filter_frames() also normalizes per-frame.
        # Keeping this for now to minimize behavior changes in the non-empty path.
        vis_np = l2norm_np(vis_np)

        # 3) Frame filtering (returns (M, D), kept_indices)
        filtered = filter_frames(vis_np)
        if isinstance(filtered, tuple):
            vis_np_filt, kept_idx = filtered
        else:
            vis_np_filt = filtered
            kept_idx = None

        vis_np_filt = np.asarray(vis_np_filt, dtype=np.float32)
        filt_n = int(vis_np_filt.shape[0])
        info["filtered_frames"] = filt_n
        info["kept_indices"] = kept_idx.tolist() if kept_idx is not None else []

        # 4) Empty-after-filter handling
        if filt_n == 0:
            info["empty_after_filter"] = True

            if self.empty_after_filter_policy == "fallback":
                # Legacy behavior (NOT recommended): keep all original frames.
                # IMPORTANT: do proper *per-row* normalization here.
                info["used_unfiltered_fallback"] = True
                V = np.asarray(vis_np, dtype=np.float32)
                row_norm = np.linalg.norm(V, axis=-1, keepdims=True)
                V = V / np.clip(row_norm, 1e-12, None)
                info["frames_used"] = int(V.shape[0])
            else:
                # Training-aligned (recommended): treat as unmatchable and do not index.
                info["skip_match_and_index"] = True
                info["frames_used"] = 0
                z0 = np.zeros((self.dim,), dtype=np.float32)
                return (z0, info) if return_info else z0
        else:
            V = vis_np_filt
            info["frames_used"] = filt_n

        orig_n = int(info["orig_frames"])
        used_n = int(info["frames_used"])
        if orig_n != used_n:
            logging.info(f"encode_entry: filtered {orig_n} → {used_n} frames")

        # 5) Attr vector
        avec = vec_from_schema(attrs or {}, self.attr_schema) if self.attr_schema else np.zeros((0,), np.float32)

        # 6) Model forward
        with torch.no_grad():
            vis_t = torch.from_numpy(V).unsqueeze(0).float()
            mask = torch.ones((1, V.shape[0]), dtype=torch.bool)
            attr_t = torch.from_numpy(avec).unsqueeze(0).float() if avec.size > 0 else None
            z = self.model(vis_t, attr_t, mask=mask).squeeze(0).cpu().numpy()

        z = l2norm_np(z.astype(np.float32))
        return (z, info) if return_info else z

    def query_then_add(
        self,
        day_id: str,
        entry_id: str,
        store_id: Optional[str] = None,
        employee_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        direction: Optional[str] = None,
        camera: Optional[str] = None,
        image_uris: Optional[List[str]] = None,
        embeddings: List[List[float]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        topk: Optional[int] = None,
        persist: bool = True,
        timezone: str = "America/New_York",
    ) -> Dict[str, Any]:
        # Ensure day_id is calculated from timestamp, ignoring any provided day_id
        day_id = _get_day_id_for_utc_timestamp(timestamp, timezone)

        resolved_store = store_id or self._store_from_entry(entry_id)

        self.total_processed += 1

        # Encode FIRST so we can early-exit on low-quality entries without rebuilding/loading FAISS.
        z, enc_info = self.encode_entry(embeddings or [], attrs, return_info=True)

        # Base decision
        decision: Dict[str, Any] = {
            "status": "new",
            "match_id": None,
            "score": None,
            "score2": None,
            # Useful for debugging parity / production tails
            "frame_filter": enc_info,
        }

        # Training skips entries that become empty after filtering; runtime should not try to match/index them.
        if enc_info.get("skip_match_and_index"):
            self.total_empty_after_filter += 1
            decision["reason"] = enc_info.get("reason") or "empty_after_filter"

            logging.info(
                "EMPTY_AFTER_FILTER entry=%s store=%s day=%s orig_frames=%s filtered_frames=%s policy=%s",
                entry_id,
                resolved_store,
                day_id,
                enc_info.get("orig_frames"),
                enc_info.get("filtered_frames"),
                enc_info.get("empty_policy"),
            )

            # optional persistence (raw inputs)
            if persist and self.db is not None:
                doc = {
                    "_id": entry_id,
                    "day_id": day_id,  # Use the calculated day_id
                    "store_id": resolved_store,
                    "alert_id": alert_id,
                    "timestamp": timestamp,
                    "direction": direction,
                    "camera": camera,
                    "images": image_uris or [],
                    "vis": {
                        "per_image_dim": len(embeddings[0]) if embeddings else 0,
                        "embeddings": embeddings,
                        "filter": enc_info,
                    },
                    "attrs": attrs or {},
                }
                if employee_id:
                    doc["employee_id"] = employee_id
                self.db[self.entries_collection].replace_one({"_id": entry_id}, doc, upsert=True)

            return decision

        # --- FAISS bucket setup (only needed when we actually try to match/index) ---
        if self.by_day_store.get(day_id, {}).get(resolved_store):
            logging.info(f"Using in-memory index for day {day_id}, store {resolved_store}.")
        elif not self.load_from_cache(day_id, resolved_store):
            self.rebuild_from_mongo(day_id)

        idx = self.ensure_day_store(day_id, resolved_store)

        # query before add (avoid matching ourselves)
        if idx.index.ntotal > 0:
            K = topk or self.topk
            sims, nn_idx, ids = idx.search(z, K)
            # NOTE: FAISS pads results with -1 indices when K > ntotal.
            # Be defensive to avoid accidentally indexing ids[-1].
            n_total = int(idx.index.ntotal)

            nn1 = float(sims[0]) if len(sims) > 0 else -1.0
            nn2 = float(sims[1]) if (n_total > 1 and len(sims) > 1) else -1.0

            nn_i1 = int(nn_idx[0]) if len(nn_idx) > 0 else -1
            nn_i2 = int(nn_idx[1]) if len(nn_idx) > 1 else -1

            id1 = ids[nn_i1] if (0 <= nn_i1 < len(ids)) else None
            id2 = ids[nn_i2] if (0 <= nn_i2 < len(ids)) else None

            gap = nn1 - nn2
            is_match = (nn1 >= self.threshold) and (gap >= self.margin)
            if is_match and id1 is not None:
                decision.update({"status": "match", "match_id": id1})
            decision.update({"score": nn1, "score2": nn2})

            # Similarity logging (essential for debugging prod behavior)
            # Default: log all MATCH decisions at INFO and borderlines near threshold.
            try:
                log_all = os.environ.get("LOG_MATCH_DECISIONS", "0").lower() in ("1", "true", "yes")
            except Exception:
                log_all = False

            should_log = log_all or (decision["status"] == "match") or (nn1 >= (self.threshold - 0.05))
            if should_log:
                ff = decision.get("frame_filter") or {}
                logging.info(
                    "MATCH_DECISION entry=%s store=%s day=%s status=%s nn1=%.4f nn2=%.4f gap=%.4f "
                    "thr=%.3f margin=%.3f topk=%d ntotal=%d match_id=%s nn2_id=%s orig_frames=%s frames_used=%s",
                    entry_id,
                    resolved_store,
                    day_id,
                    decision["status"],
                    nn1,
                    nn2,
                    gap,
                    self.threshold,
                    self.margin,
                    int(K),
                    n_total,
                    decision.get("match_id"),
                    id2,
                    ff.get("orig_frames"),
                    ff.get("frames_used"),
                )

        # add to FAISS
        idx.add(z, entry_id)

        # save to cache after adding
        self.save_to_cache(day_id, resolved_store)

        # optional persistence (raw inputs)
        if persist and self.db is not None:
            doc = {
                "_id": entry_id,
                "day_id": day_id,  # Use the calculated day_id
                "store_id": resolved_store,
                "alert_id": alert_id,
                "timestamp": timestamp,
                "direction": direction,
                "camera": camera,
                "images": image_uris or [],
                "vis": {
                    "per_image_dim": len(embeddings[0]) if embeddings else 0,
                    "embeddings": embeddings,
                    "filter": enc_info,
                },
                "attrs": attrs or {},
            }
            if employee_id:
                doc["employee_id"] = employee_id
            self.db[self.entries_collection].replace_one({"_id": entry_id}, doc, upsert=True)

        return decision

    def rebuild_from_mongo(self, day_id: str) -> Dict[str, Any]:
        logging.info(f"Rebuilding index for day {day_id} from Mongo.")
        if self.db is None:
            return {"ok": False, "error": "Mongo not configured"}
        self.by_day_store[day_id] = {}
        cur = self.db[self.entries_collection].find(
            {"day_id": day_id},
            {"_id": 1, "store_id": 1, "vis.embeddings": 1, "attrs": 1, "images": 1},
        )
        cnt = 0
        skipped_empty = 0
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

            z, enc_info = self.encode_entry(embs, attrs, return_info=True)
            if enc_info.get("skip_match_and_index"):
                skipped_empty += 1
                continue

            idx.add(z, eid)
            cnt += 1

        for store_id in stores_rebuilt:
            self.save_to_cache(day_id, store_id)

        return {"ok": True, "count": cnt, "skipped_empty_after_filter": skipped_empty}

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

    def build_manifest(self, day_id: str, store_id: str, entry_id: Optional[str] = None, cameras: Optional[List[str]] = None, timezone: str = "America/New_York") -> Dict[str, Any]:
        if self.db is None:
            raise ValueError("Mongo not configured")

        # Convert day_id to a local timezone-aware date range, then to UTC for querying
        try:
            local_tz = pytz.timezone(timezone)
            start_of_day_local = local_tz.localize(datetime.strptime(day_id, "%Y-%m-%d"))
            end_of_day_local = start_of_day_local + timedelta(days=1)
            start_of_day_utc = start_of_day_local.astimezone(pytz.utc)
            end_of_day_utc = end_of_day_local.astimezone(pytz.utc)
        except Exception as e:
            raise ValueError(f"Invalid day_id or timezone: {e}")

        # Check if index is in memory, if not, try to load from cache or rebuild
        if self.by_day_store.get(day_id, {}).get(store_id):
            logging.info(f"Using in-memory index for day {day_id}, store {store_id}.")
        elif not self.load_from_cache(day_id, store_id):
            self.rebuild_from_mongo(day_id)

        query = {
            "timestamp": {
                "$gte": start_of_day_utc.isoformat(),
                "$lt": end_of_day_utc.isoformat(),
            },
            "store_id": store_id,
        }
        if cameras:
            query["camera"] = {"$in": cameras}
        docs = list(self.db[self.entries_collection].find(query))
        if not docs:
            return {"day_id": day_id, "store_id": store_id, "person_count": 0, "event_count": 0, "people": []}
        entry_ids = [doc["_id"] for doc in docs]
        meta_map = self._load_event_metadata(entry_ids)
        records = []
        for doc in docs:
            eid = doc["_id"]
            if eid == "nc0009-salefloor-270-556b749d-44":
                logging.info(f"DEBUG: Processing entry {eid}")
                logging.info(f"DEBUG: doc from observations: {doc}")
                meta = meta_map.get(eid, {})
                logging.info(f"DEBUG: meta from historian: {meta}")

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
                    "employee_id": doc.get("employee_id"),
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
        added_ids: List[str] = []
        people: List[Dict[str, Any]] = []


        for rec in records:
            vec, enc_info = self.encode_entry(rec["embeddings"], rec["attrs"], return_info=True)

            cluster_idx = None
            nn1 = None
            nn2 = None
            match_id = None

            # Only attempt to match/index if the entry is matchable.
            if (not enc_info.get("skip_match_and_index")) and temp_index.ntotal > 0:
                K = min(temp_index.ntotal, max(2, self.topk))
                sims, idxs = temp_index.search(vec.reshape(1, -1).astype(np.float32), K)
                nn1 = float(sims[0][0])
                nn2 = float(sims[0][1]) if K > 1 else -1.0
                is_match = (nn1 >= self.threshold) and ((nn1 - nn2) >= self.margin)
                if is_match:
                    hit_idx = int(idxs[0][0])
                    cluster_idx = assignments[hit_idx]
                    match_id = added_ids[hit_idx]

            if cluster_idx is None:
                cluster_idx = len(people)
                first_seen = rec["timestamp"]
                people.append(
                    {"person_id": f"{store_id}-{cluster_idx+1:04d}", "first_seen": first_seen, "events": []}
                )

            # Attach a minimal filter summary for debugging
            frame_filter_summary = {
                "orig_frames": enc_info.get("orig_frames"),
                "filtered_frames": enc_info.get("filtered_frames"),
                "frames_used": enc_info.get("frames_used"),
                "empty_after_filter": enc_info.get("empty_after_filter"),
                "empty_policy": enc_info.get("empty_policy"),
                "used_unfiltered_fallback": enc_info.get("used_unfiltered_fallback"),
            }

            people[cluster_idx]["events"].append(
                {
                    "entry_id": rec["entry_id"],
                    "timestamp": rec["timestamp"],
                    "direction": rec["direction"],
                    "camera": rec["camera"],
                    "alert_id": rec["alert_id"],
                    "employee_id": rec.get("employee_id"),
                    "images": rec["images"],
                    "score": nn1,
                    "score2": nn2,
                    "attrs": rec["attrs"],
                    "embeddings": rec["embeddings"],
                    "frame_filter": frame_filter_summary,
                    "match_decision": {
                        "status": "match" if match_id else ("skip" if enc_info.get("skip_match_and_index") else "new"),
                        "match_id": match_id,
                        "score": nn1,
                        "score2": nn2,
                        "frame_filter": frame_filter_summary,
                    },
                }
            )

            if rec["timestamp"]:
                if (
                    people[cluster_idx]["first_seen"] is None
                    or self._ts_key(rec["timestamp"]) < self._ts_key(people[cluster_idx]["first_seen"])
                ):
                    people[cluster_idx]["first_seen"] = rec["timestamp"]

            # Only index matchable entries (avoid poisoning the candidate set with low-quality events)
            if not enc_info.get("skip_match_and_index"):
                temp_index.add(vec.reshape(1, -1).astype(np.float32))
                assignments.append(cluster_idx)
                added_ids.append(rec["entry_id"])

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

    def build_employee_manifest(self, day_id: str, store_id: str, emp_id: str) -> Dict[str, Any]:
        full_manifest = self.build_manifest(day_id, store_id)

        employee_person_ids = set()
        for person in full_manifest.get("people", []):
            for event in person.get("events", []):
                if event.get("employee_id") == emp_id:
                    employee_person_ids.add(person["person_id"])
                    break  # Found a match in this cluster, no need to check other events

        employee_clusters = [p for p in full_manifest.get("people", []) if p["person_id"] in employee_person_ids]

        # Define external doors and modify the event payload
        external_doors = {
            'nc0009-back-door',
            'nc0009-salefloor-270',
            'nc0211-front-door-1',
            'nc0211-front-door-2',
            'nc0211-safe-back-door-270'
        }
        for person in employee_clusters:
            for event in person.get("events", []):
                event.pop("attrs", None)
                event.pop("embeddings", None)
                event["external_door"] = event.get("camera") in external_doors

        return {
            "day_id": day_id,
            "store_id": store_id,
            "employee_id": emp_id,
            "person_count": len(employee_clusters),
            "event_count": sum(len(p.get("events", [])) for p in employee_clusters),
            "people": employee_clusters,
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

        vec_a, info_a = self.encode_entry(embs_a, attrs_a, return_info=True)
        vec_b, info_b = self.encode_entry(embs_b, attrs_b, return_info=True)
        score = float(np.dot(vec_a, vec_b))

        return {
            "score": score,
            "frame_filter_a": {
                "orig_frames": info_a.get("orig_frames"),
                "filtered_frames": info_a.get("filtered_frames"),
                "frames_used": info_a.get("frames_used"),
                "empty_after_filter": info_a.get("empty_after_filter"),
                "empty_policy": info_a.get("empty_policy"),
                "used_unfiltered_fallback": info_a.get("used_unfiltered_fallback"),
                "skip_match_and_index": info_a.get("skip_match_and_index"),
            },
            "frame_filter_b": {
                "orig_frames": info_b.get("orig_frames"),
                "filtered_frames": info_b.get("filtered_frames"),
                "frames_used": info_b.get("frames_used"),
                "empty_after_filter": info_b.get("empty_after_filter"),
                "empty_policy": info_b.get("empty_policy"),
                "used_unfiltered_fallback": info_b.get("used_unfiltered_fallback"),
                "skip_match_and_index": info_b.get("skip_match_and_index"),
            },
            "entry_a": self._doc_to_event(doc_a),
            "entry_b": self._doc_to_event(doc_b),
        }
