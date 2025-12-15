"""
Start:
    export PAIRS_JSONL=pairs_hard.jsonl
    export MANIFEST_JSON=manifest.json
    uvicorn gt.pair_viewer.server:app --host 0.0.0.0 --port 9001 --reload

Then open:
    http://localhost:9001/static/index.html
"""
from __future__ import annotations

import importlib.util
import json
import logging
import mimetypes
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional
from urllib.parse import quote_plus

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient

logger = logging.getLogger(__name__)

# Resolve helpers with an explicit package context so relative imports work.
REPO_ROOT = Path(__file__).resolve().parents[2]
PAIR_VIEWER_DIR = Path(__file__).resolve().parent
MANIFEST_EDITOR_DIR = PAIR_VIEWER_DIR.parent

for path in (REPO_ROOT, MANIFEST_EDITOR_DIR, PAIR_VIEWER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Create a synthetic package so manifest-editor modules can use relative imports
pkg_name = "manifest_editor"
pkg = sys.modules.get(pkg_name)
if pkg is None:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(MANIFEST_EDITOR_DIR)]
    sys.modules[pkg_name] = pkg

def _load_module(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load module {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

config_mod = _load_module(f"{pkg_name}.config", MANIFEST_EDITOR_DIR / "config.py")
gcs_mod = _load_module(f"{pkg_name}.gcs", MANIFEST_EDITOR_DIR / "gcs.py")
download_blob_bytes = gcs_mod.download_blob_bytes  # type: ignore
settings = config_mod.get_settings()  # type: ignore

BASE_URL = os.getenv("BASE_URL", "/").rstrip("/")
PAIRS_VIEWER_BASE = f"{BASE_URL}/pairs-viewer"
if not PAIRS_VIEWER_BASE.startswith("/"):
    PAIRS_VIEWER_BASE = "/" + PAIRS_VIEWER_BASE
MANIFEST_EDITOR_IMAGE_API = f"{BASE_URL}/manifest-editor/api/image"
if not MANIFEST_EDITOR_IMAGE_API.startswith("/"):
    MANIFEST_EDITOR_IMAGE_API = "/" + MANIFEST_EDITOR_IMAGE_API

# MongoDB Setup
# Always target gt_tools for pairs viewer writes, regardless of manifest editor DB
MONGO_URI = os.getenv("MANIFEST_EDITOR_MONGO", "mongodb://teknoir:change-me@localhost:37017")
DB_NAME = "gt_tools"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
clusters_coll = db["clusters"]
reviewed_pairs_coll = db["reviewed_pairs"]
try:
    _REVIEWED_SET = {doc["_id"] for doc in reviewed_pairs_coll.find({}, {"_id": 1})}
    print(f"[init] loaded {len(_REVIEWED_SET)} reviewed pairs from Mongo")
except Exception as exc:  # pragma: no cover
    _REVIEWED_SET: set[str] = set()
    print(f"[warn] could not load reviewed pairs: {exc}")


app = FastAPI(title="GT High‑risk Pairs Viewer", version="1.0")

PAIRS_JSONL = Path(os.environ.get("PAIRS_JSONL", "pairs_hard.jsonl"))
MANIFEST_JSON = Path(os.environ.get("MANIFEST_JSON", "manifest.json"))

def _pair_hash(entry_id_a: str, entry_id_b: str) -> str:
    return "_".join(sorted([entry_id_a, entry_id_b]))


class PairOut(BaseModel):
    idx: int
    risk_type: str
    sim: float
    entry_id_a: str
    entry_id_b: str
    pid_a: str
    pid_b: str
    same_pid: int
    store_a: Optional[str] = None
    store_b: Optional[str] = None
    day_a: Optional[str] = None
    day_b: Optional[str] = None
    camera_a: Optional[str] = None
    camera_b: Optional[str] = None
    ts_a: Optional[str] = None
    ts_b: Optional[str] = None
    image_a: Optional[str] = None
    image_b: Optional[str] = None
    images_a: List[str] = []
    images_b: List[str] = []


def _resolve_image_uri(uri: str) -> str:
    """Normalize incoming image identifiers into fully-qualified URIs."""
    if uri.startswith(("gs://", "http://", "https://")):
        return uri
    if uri.startswith("media/lc-person-cutouts"):
        return f"gs://victra-poc.teknoir.cloud/{uri.lstrip('/')}"
    if uri.startswith("media/"):
        return f"gs://computer_vision_model_zoo/{uri.lstrip('/')}"
    return uri


def _rewrite_image(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return uri
    uri = _resolve_image_uri(uri)
    if uri.startswith("gs://"):
        # Use manifest editor's image proxy so we stay consistent with GT browser
        return f"{MANIFEST_EDITOR_IMAGE_API}?source={quote_plus(uri)}"
    return uri


def _entry_images(entry: Optional[dict], fallback: Optional[str] = None) -> List[str]:
    sources: List[str] = []
    if entry:
        raw = entry.get("files") or entry.get("images") or entry.get("image_paths") or []
        sources = [_rewrite_image(img) for img in raw if img]
    if not sources and fallback:
        fb = _rewrite_image(fallback)
        if fb:
            sources = [fb]
    return sources


def load_pairs() -> List[dict]:
    if not PAIRS_JSONL.exists():
        raise RuntimeError(f"PAIRS_JSONL file not found: {PAIRS_JSONL}")
    out: List[dict] = []
    with PAIRS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # rewrite image URIs if BUCKET_PREFIX is set
            obj["image_a"] = _rewrite_image(obj.get("image_a"))
            obj["image_b"] = _rewrite_image(obj.get("image_b"))
            out.append(obj)
    return out

def load_manifest() -> Dict[str, dict]:
    if not MANIFEST_JSON.exists():
        raise RuntimeError(f"MANIFEST_JSON file not found: {MANIFEST_JSON}")
    return json.loads(MANIFEST_JSON.read_text())


# load once at startup
try:
    _PAIRS_CACHE: List[dict] = load_pairs()
    print(f"[init] loaded {len(_PAIRS_CACHE)} pairs from {PAIRS_JSONL}")
except Exception as exc:  # pragma: no cover
    _PAIRS_CACHE = []
    print(f"[warn] could not load pairs from {PAIRS_JSONL}: {exc}")

try:
    _MANIFEST_CACHE: Dict[str, dict] = load_manifest()
    print(f"[init] loaded {len(_MANIFEST_CACHE)} entries from {MANIFEST_JSON}")
except Exception as exc:  # pragma: no cover
    _MANIFEST_CACHE = {}
    print(f"[warn] could not load manifest from {MANIFEST_JSON}: {exc}")


@app.post("/exclude_person")
def exclude_person(
    store_id: str = Query(..., description="Store ID"),
    day_id: str = Query(..., description="Day ID"),
    person_id: str = Query(..., description="Person ID to exclude"),
    entry_id: Optional[str] = Query(None, description="Entry ID to resolve person_id from clusters"),
):
    try:
        resolved_pid = person_id
        if entry_id:
            cluster = clusters_coll.find_one(
                {"store_id": store_id, "day_id": day_id}, {"persons": 1}
            )
            if not cluster:
                raise HTTPException(status_code=404, detail="Cluster not found")
            persons = cluster.get("persons") or {}
            found_pid = None
            for pid, person in persons.items():
                for ev in person.get("entries", []):
                    if ev.get("entry_id") == entry_id:
                        found_pid = pid
                        break
                if found_pid:
                    break
            if not found_pid:
                raise HTTPException(
                    status_code=404,
                    detail="Person containing entry_id not found in cluster",
                )
            resolved_pid = found_pid

        # Construct the field to update dynamically
        field_to_set = f"persons.{resolved_pid}.include"
        result = clusters_coll.update_one(
            {"store_id": store_id, "day_id": day_id},
            {"$set": {field_to_set: False}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Cluster not found")
        return {
            "status": "success",
            "resolved_pid": resolved_pid,
            "message": f"Person {resolved_pid} in {store_id}-{day_id} excluded.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mark_reviewed")
def mark_reviewed(
    entry_id_a: str = Query(..., description="First Entry ID in the pair"),
    entry_id_b: str = Query(..., description="Second Entry ID in the pair"),
):
    try:
        # Ensure consistent ordering for the pair ID
        pair_hash = _pair_hash(entry_id_a, entry_id_b)
        result = reviewed_pairs_coll.update_one(
            {"_id": pair_hash},
            {"$set": {"timestamp": datetime.now().isoformat(), "entry_a": entry_id_a, "entry_b": entry_id_b}},
            upsert=True
        )
        _REVIEWED_SET.add(pair_hash)
        return {"status": "success", "message": f"Pair {pair_hash} marked as reviewed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"ok": True, "pairs": len(_PAIRS_CACHE), "manifest_entries": len(_MANIFEST_CACHE)}


@app.get("/stores", response_model=List[str])
def get_stores():
    if not _PAIRS_CACHE:
        return []
    stores = sorted(list(set(r.get("store_a") for r in _PAIRS_CACHE if r.get("store_a"))))
    return stores


@app.get("/days", response_model=List[str])
def get_days():
    if not _PAIRS_CACHE:
        return []
    days = sorted(list(set(r.get("day_a") for r in _PAIRS_CACHE if r.get("day_a"))))
    return days


@app.get("/pairs", response_model=List[PairOut])
def list_pairs(
    risk_type: Optional[Literal["hard_neg", "hard_pos", "alias_candidate"]] = Query(
        default=None, description="Filter by risk_type"
    ),
    store_id: Optional[str] = Query(default=None, description="Filter by store ID"),
    day_id: Optional[str] = Query(default=None, description="Filter by day ID"),
    include_reviewed: bool = Query(default=False, description="Include pairs already marked reviewed"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sort: Literal["sim_desc", "sim_asc"] = Query(
        default="sim_desc", description="sort by similarity"
    ),
    ts_sort: Optional[Literal["ts_asc", "ts_desc"]] = Query(
        default=None, description="sort by timestamp of Entry A"
    ),
    store_sort: Optional[Literal["store_asc", "store_desc"]] = Query(
        default=None, description="sort by store of Entry A"
    ),
):
    if not _PAIRS_CACHE:
        raise HTTPException(status_code=404, detail="No pairs loaded")

    rows = _PAIRS_CACHE
    if not include_reviewed and _REVIEWED_SET:
        rows = [
            r
            for r in rows
            if _pair_hash(r.get("entry_id_a", ""), r.get("entry_id_b", "")) not in _REVIEWED_SET
        ]
    if risk_type:
        rows = [r for r in rows if r.get("risk_type") == risk_type]
    if store_id:
        rows = [r for r in rows if r.get("store_a") == store_id]
    if day_id:
        rows = [r for r in rows if r.get("day_a") == day_id]

    if store_sort:
        reverse_store = store_sort == "store_desc"
        rows = sorted(rows, key=lambda r: r.get("store_a") or "", reverse=reverse_store)
    elif ts_sort:
        reverse_ts = ts_sort == "ts_desc"
        rows = sorted(rows, key=lambda r: r.get("ts_a") or "", reverse=reverse_ts)
    else:  # Default to sorting by similarity if no other sort is specified
        reverse_sim = sort == "sim_desc"
        rows = sorted(rows, key=lambda r: float(r.get("sim", 0.0)), reverse=reverse_sim)

    sliced = rows[offset : offset + limit]
    out: List[PairOut] = []
    for idx, r in enumerate(sliced):
        entry_a = _MANIFEST_CACHE.get(r.get("entry_id_a"))
        entry_b = _MANIFEST_CACHE.get(r.get("entry_id_b"))

        images_a = _entry_images(entry_a, r.get("image_a"))
        images_b = _entry_images(entry_b, r.get("image_b"))

        out.append(
            PairOut(
                idx=idx + offset,
                risk_type=r.get("risk_type", ""),
                sim=float(r.get("sim", 0.0)),
                entry_id_a=r.get("entry_id_a", ""),
                entry_id_b=r.get("entry_id_b", ""),
                pid_a=str(r.get("pid_a")),
                pid_b=str(r.get("pid_b")),
                same_pid=int(r.get("same_pid", 0)),
                store_a=r.get("store_a"),
                store_b=r.get("store_b"),
                day_a=r.get("day_a"),
                day_b=r.get("day_b"),
                camera_a=r.get("camera_a"),
                camera_b=r.get("camera_b"),
                ts_a=r.get("ts_a"),
                ts_b=r.get("ts_b"),
                image_a=r.get("image_a"),
                image_b=r.get("image_b"),
                images_a=images_a,
                images_b=images_b,
            )
        )
    return out


@app.get("/pairs/{idx}", response_model=PairOut)
def get_pair(idx: int):
    if not _PAIRS_CACHE:
        raise HTTPException(status_code=404, detail="No pairs loaded")
    if idx < 0 or idx >= len(_PAIRS_CACHE):
        raise HTTPException(status_code=404, detail="Index out of range")
    r = _PAIRS_CACHE[idx]

    entry_a = _MANIFEST_CACHE.get(r.get("entry_id_a"))
    entry_b = _MANIFEST_CACHE.get(r.get("entry_id_b"))

    images_a = _entry_images(entry_a, r.get("image_a"))
    images_b = _entry_images(entry_b, r.get("image_b"))

    return PairOut(
        idx=idx,
        risk_type=r.get("risk_type", ""),
        sim=float(r.get("sim", 0.0)),
        entry_id_a=r.get("entry_id_a", ""),
        entry_id_b=r.get("entry_id_b", ""),
        pid_a=str(r.get("pid_a")),
        pid_b=str(r.get("pid_b")),
        same_pid=int(r.get("same_pid", 0)),
        store_a=r.get("store_a"),
        store_b=r.get("store_b"),
        day_a=r.get("day_a"),
        day_b=r.get("day_b"),
        camera_a=r.get("camera_a"),
        camera_b=r.get("camera_b"),
        ts_a=r.get("ts_a"),
        ts_b=r.get("ts_b"),
        image_a=r.get("image_a"),
        image_b=r.get("image_b"),
        images_a=images_a,
        images_b=images_b,
    )

@app.get("/entry/{entry_id}")
def get_entry(entry_id: str):
    if not _MANIFEST_CACHE:
        raise HTTPException(status_code=404, detail="Manifest not loaded")
    entry = _MANIFEST_CACHE.get(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    images = _entry_images(entry)
    return {**entry, "images": images}


@app.get("/image")
def image_proxy(source: str = Query(..., description="gs:// URI or remote path to image")):
    uri = _resolve_image_uri(source)
    try:
        if uri.startswith("gs://"):
            data = download_blob_bytes(uri)
        else:
            resp = requests.get(uri, timeout=30)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch image: {uri}")
            data = resp.content
        mime = mimetypes.guess_type(uri)[0] or "image/jpeg"
        return StreamingResponse(iter([data]), media_type=mime)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to fetch image: {exc}") from exc

import subprocess

@app.post("/update")
def update_pairs():
    """Run the pair generation script and reload the data."""
    model_run = "xattn_pk_64_2_v5"
    output_dir = "gt"
    pairs_jsonl_path = Path(output_dir) / "pairs_hard.jsonl"

    # The MANIFEST_JSON path also needs to be located relative to the script run
    # Assuming it's in the same base directory structure
    manifest_path = f"model/encoder/runs/entry.json"

    command = f"""
        python3 gt/find_high_risk_pairs.py \
          --vecs model/encoder/runs/{model_run}/entry_vectors.npy \
          --ids model/encoder/runs/{model_run}/entry_ids.txt \
          --manifest {manifest_path} \
          --gt model/encoder/runs/multi_gt.json \
          --neg-threshold 0.86 \
          --pos-threshold 0.70 \
          --topk-neg 20 \
          --output-dir {output_dir}
    """
    try:
        print("Running update script...")
        process = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=ROOT  # Run from the project root directory
        )
        print(process.stdout)
        if process.stderr:
            print(process.stderr, file=sys.stderr)

        # Reload the pairs into memory
        global _PAIRS_CACHE, MANIFEST_JSON
        # Update environment or global vars to point to new files for reloading
        os.environ["PAIRS_JSONL"] = str(pairs_jsonl_path)
        os.environ["MANIFEST_JSON"] = manifest_path
        PAIRS_JSONL = pairs_jsonl_path
        MANIFEST_JSON = Path(manifest_path)

        _PAIRS_CACHE = load_pairs()
        print(f"[reload] loaded {len(_PAIRS_CACHE)} pairs from {PAIRS_JSONL}")

        return {
            "status": "success",
            "message": f"Successfully updated pairs. Loaded {len(_PAIRS_CACHE)} new pairs.",
            "log": process.stdout
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to run update script.",
                "stdout": e.stdout,
                "stderr": e.stderr,
            },
        )

app.mount("/static", StaticFiles(directory=Path(__file__).parent.resolve()), name="static")
