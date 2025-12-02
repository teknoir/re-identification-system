from __future__ import annotations

import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient

from .gcs import download_blob_bytes 

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = Path(__file__).with_name("manifest_editor.html")
API_EDITOR_PATH = Path(__file__).with_name("manifest_api_editor.html")
ENCODER_VIEWER_PATH = Path(__file__).with_name("encoder_dataset_viewer.html")
GT_EDITOR_PATH = Path(__file__).with_name("gt_editor.html")
ENCODER_STATIC_DIR = ROOT / "model" / "encoder" / "runs"
STATE_PATH = Path(os.getenv("MANIFEST_EDITOR_STATE", ROOT / "gt/manifest_editor_state.json"))
MANIFEST_API_BASE = os.getenv("MANIFEST_API_BASE", "http://matching-service")
MONGO_URI = os.getenv("MANIFEST_EDITOR_MONGO", "mongodb://teknoir:change-me@localhost:37017")
MONGO_DB = os.getenv("MANIFEST_EDITOR_DB", "gt_tools")
ENTRIES_COLL = os.getenv("MANIFEST_EDITOR_ENTRIES_COLL", "entries")
CLUSTERS_COLL = os.getenv("MANIFEST_EDITOR_CLUSTERS_COLL", "clusters")
GT_COLL = os.getenv("MANIFEST_EDITOR_GT_COLL", "map")
BLOB_BASE = os.getenv("MANIFEST_EDITOR_BUCKET", "gs://victra-poc.teknoir.cloud")  # optional prefix for relative files
BASE_URL =  os.getenv("BASE_URL", "/")  # optional base url
MANIFEST_API_TIMEOUT_SECONDS = int(os.getenv("MANIFEST_API_TIMEOUT_SECONDS", "60"))

import sys  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

 # noqa: E402

app = FastAPI(root_path=os.path.join(BASE_URL, "manifest-editor"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_client: Optional[MongoClient] = None


def get_db():
    global mongo_client
    if mongo_client is None:
        mongo_client = MongoClient(MONGO_URI)
    return mongo_client[MONGO_DB]


def load_manifest(source: str) -> Dict[str, Any]:
    raise HTTPException(status_code=400, detail="Manifest loading from arbitrary sources is disabled in this build.")


def iter_people(manifest: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    people = manifest.get("people")
    if isinstance(people, list):
        return people
    if isinstance(people, dict):
        return people.values()
    return manifest.get("persons", {}).values() if isinstance(manifest.get("persons"), dict) else []


def normalize_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    for person in iter_people(manifest):
        events = person.get("events") or person.get("entries") or []
        for ev in events:
            images = ev.get("images") or []
            ev["_image_paths"] = list(images)
    return manifest


def upsert_entries(entries: List[Dict[str, Any]]) -> List[str]:
    db = get_db()
    coll = db[ENTRIES_COLL]
    now = datetime.utcnow()
    seen: List[str] = []
    for ev in entries:
        entry_id = ev.get("entry_id")
        if not entry_id:
            continue
        seen.append(entry_id)
        files = ev.get("files") or ev.get("images") or []
        if BLOB_BASE:
            files = [
                f if f.startswith(("gs://", "http://", "https://")) else BLOB_BASE.rstrip("/") + "/" + f.lstrip("/")
                for f in files
            ]
        doc = {
            "_id": entry_id,
            "entry_id": entry_id,
            "store_id": ev.get("store_id"),
            "day_id": ev.get("day_id"),
            "camera": ev.get("camera"),
            "timestamp": ev.get("timestamp"),
            "direction": ev.get("direction"),
            "alert_id": ev.get("alert_id"),
            "files": files,
            "attrs": ev.get("attrs") or ev.get("attr") or {},
            "embeddings": ev.get("embeddings"),
            "meta": {"saved_at": now},
        }
        coll.replace_one({"_id": entry_id}, doc, upsert=True)
    return seen


def upsert_cluster(store_id: str, day_id: str, persons_map: Dict[str, Any], adjudicated: bool = False):
    db = get_db()
    coll = db[CLUSTERS_COLL]
    key = f"{store_id}-{day_id}"
    doc = {
        "_id": key,
        "store_id": store_id,
        "day_id": day_id,
        "persons": persons_map,
        "meta": {"saved_at": datetime.utcnow()},
        "adjudicated": bool(adjudicated),
    }
    coll.replace_one({"_id": key}, doc, upsert=True)


def normalize_flag_value(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("remove"))
    return bool(value)


def read_manifest_from_mongo(store_id: str, day_id: str) -> Dict[str, Any]:
    db = get_db()
    clusters = db[CLUSTERS_COLL]
    entries_coll = db[ENTRIES_COLL]
    key = f"{store_id}-{day_id}"
    cluster = clusters.find_one({"_id": key}) or clusters.find_one({"store_id": store_id, "day_id": day_id})
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found in Mongo for store/day")
    persons_raw = cluster.get("persons") or {}
    entry_ids: List[str] = []
    for person in persons_raw.values():
        for ev in person.get("entries", []):
            eid = ev.get("entry_id")
            if eid:
                entry_ids.append(eid)
    entries: Dict[str, Any] = {}
    if entry_ids:
        for doc in entries_coll.find({"_id": {"$in": entry_ids}}):
            entries[doc["_id"]] = doc

    people_out = []
    for pid, person in persons_raw.items():
        events_out = []
        for ev in person.get("entries", []):
            eid = ev.get("entry_id")
            ref = entries.get(eid, {})
            images = ev.get("images") or ref.get("files") or ev.get("image_paths") or []
            events_out.append(
                {
                    "entry_id": eid,
                    "alert_id": ev.get("alert_id") or ref.get("alert_id"),
                    "store_id": ev.get("store_id") or ref.get("store_id") or store_id,
                    "day_id": ev.get("day_id") or ref.get("day_id") or day_id,
                    "camera": ev.get("camera") or ref.get("camera"),
                    "timestamp": ev.get("timestamp") or ref.get("timestamp"),
                    "direction": ev.get("direction") or ref.get("direction"),
                    "images": images,
                    "attrs": ev.get("attrs") or ref.get("attrs") or {},
                    "embeddings_ref": ev.get("embeddings_ref") or ref.get("embeddings_ref"),
                }
            )
        people_out.append(
            {
                "person_id": pid,
                "alias": person.get("alias") or "",
                "first_seen": person.get("first_seen") or "",
                "exclude": not bool(person.get("include", True)),
                "events": events_out,
            }
        )
    return {
        "ok": True,
        "source": "mongo",
        "store_id": store_id,
        "day_id": day_id,
        "adjudicated": bool(cluster.get("adjudicated", False)),
        "people": people_out,
    }


def read_editor_state() -> Dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return None


def write_editor_state(payload: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2))


@app.get("/", response_class=HTMLResponse)
def index():
    if not API_EDITOR_PATH.exists():
        raise HTTPException(status_code=500, detail="manifest_api_editor.html not found")
    return API_EDITOR_PATH.read_text(encoding="utf-8")


@app.get("/manifest_editor", response_class=HTMLResponse)
def manifest_api_editor():
    if not API_EDITOR_PATH.exists():
        raise HTTPException(status_code=500, detail="manifest_api_editor.html not found")
    return API_EDITOR_PATH.read_text(encoding="utf-8")


@app.get("/encoder_viewer", response_class=HTMLResponse)
def encoder_viewer():
    if not ENCODER_VIEWER_PATH.exists():
        raise HTTPException(status_code=500, detail="encoder_dataset_viewer.html not found")
    return ENCODER_VIEWER_PATH.read_text(encoding="utf-8")


@app.get("/gt_editor", response_class=HTMLResponse)
def gt_editor():
    if not GT_EDITOR_PATH.exists():
        raise HTTPException(status_code=500, detail="gt_editor.html not found")
    return GT_EDITOR_PATH.read_text(encoding="utf-8")

# Serve encoder run artifacts (large files) for viewer defaults
if ENCODER_STATIC_DIR.exists():
    app.mount("/static/encoder", StaticFiles(directory=str(ENCODER_STATIC_DIR)), name="encoder_static")

def _resolve_image_uri(source: str) -> str:
    if source.startswith(("gs://", "http://", "https://")):
        return source
    if BLOB_BASE:
        return f"{BLOB_BASE.rstrip('/')}/{source.lstrip('/')}"
    raise HTTPException(status_code=404, detail="Unable to resolve image path; set MANIFEST_EDITOR_BUCKET for relative paths")


@app.get("/api/image")
def image_proxy(source: str = Query(..., description="gs:// URI or remote path to image")):
    uri = _resolve_image_uri(source)
    if uri.startswith("gs://"):
        data = download_blob_bytes(uri)
    else:
        resp = requests.get(uri, timeout=30)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch remote image: {uri}")
        data = resp.content
    mime = mimetypes.guess_type(uri)[0] or "image/jpeg"
    return StreamingResponse(iter([data]), media_type=mime)


@app.get("/api/manifest-proxy")
def manifest_proxy(
    day_id: str,
    store_id: str,
    entry_id: Optional[str] = None,
    camera: Optional[List[str]] = Query(None),
):
    # First try Mongo; fall back to API if not found
    try:
        mongo_data = read_manifest_from_mongo(store_id, day_id)
        return mongo_data
    except Exception:
        pass

    base = MANIFEST_API_BASE
    params = [("day_id", day_id), ("store_id", store_id)]
    if entry_id:
        params.append(("entry_id", entry_id))
    if camera:
        for cam in camera:
            params.append(("camera", cam))
    url = f"{base.rstrip('/')}/manifest"
    print(f"[MANIFEST-EDITOR] Requesting manifest from: {url} with timeout: {MANIFEST_API_TIMEOUT_SECONDS}s", flush=True)
    resp = requests.get(url, params=params, timeout=MANIFEST_API_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    data["source"] = "api"
    return data


@app.get("/api/gt")
def get_ground_truth(store_id: str, day_id: str):
    db = get_db()
    gt_coll = db[GT_COLL]
    entries_coll = db[ENTRIES_COLL]
    key = f"{store_id}-{day_id}"
    doc = gt_coll.find_one({"_id": key}) or gt_coll.find_one({"store_id": store_id, "day_id": day_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Ground truth not found for store/day")
    entry_map = doc.get("entry_person_map") or {}
    persons_meta = doc.get("persons") or {}
    entry_ids = list(entry_map.keys())
    entries: Dict[str, Any] = {}
    if entry_ids:
        for entry_doc in entries_coll.find({"_id": {"$in": entry_ids}}):
            entries[entry_doc["_id"]] = entry_doc
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry_id, info in entry_map.items():
        pid = info.get("person_id")
        if not pid:
            continue
        meta = persons_meta.get(pid, {})
        person = grouped.setdefault(
            pid,
            {
                "person_id": pid,
                "alias": meta.get("alias") or "",
                "first_seen": meta.get("first_seen") or "",
                "entries": [],
            },
        )
        entry_doc = entries.get(entry_id) or {}
        include = bool(info.get("include", True))
        person["entries"].append(
            {
                "entry_id": entry_id,
                "store_id": entry_doc.get("store_id") or store_id,
                "day_id": entry_doc.get("day_id") or day_id,
                "camera": entry_doc.get("camera"),
                "timestamp": entry_doc.get("timestamp"),
                "direction": entry_doc.get("direction"),
                "images": entry_doc.get("files") or entry_doc.get("images") or [],
                "attrs": entry_doc.get("attrs") or {},
                "flagged": not include,
            }
        )
    people_out = []
    for pid in sorted(grouped.keys()):
        people_out.append(grouped[pid])
    return {
        "ok": True,
        "store_id": doc.get("store_id") or store_id,
        "day_id": doc.get("day_id") or day_id,
        "people": people_out,
        "meta": doc.get("meta") or {},
    }


@app.post("/api/gt/flags")
def update_gt_flags(payload: Dict[str, Any] = Body(...)):
    store_id = payload.get("store_id")
    day_id = payload.get("day_id")
    if not store_id or not day_id:
        raise HTTPException(status_code=400, detail="store_id and day_id are required")
    flags = payload.get("flags") or {}
    removals = payload.get("remove_entries") or []
    if not isinstance(flags, dict):
        raise HTTPException(status_code=400, detail="flags must be an object of entry_id -> bool")
    db = get_db()
    gt_coll = db[GT_COLL]
    key = f"{store_id}-{day_id}"
    doc = gt_coll.find_one({"_id": key})
    if not doc:
        raise HTTPException(status_code=404, detail="Ground truth document not found")
    entry_map = doc.get("entry_map") or {}
    set_map = {}
    for entry_id, flag in flags.items():
        if entry_id not in entry_map:
            continue
        set_map[f"entry_map.{entry_id}.include"] = not bool(flag)
    if not set_map:
        return {"ok": True, "updated": 0, "removed": 0}
    result = gt_coll.update_one({"_id": key}, {"$set": set_map})
    removed = sum(1 for v in flags.values() if v)
    return {"ok": True, "updated": int(result.modified_count), "removed": removed}


@app.get("/api/gt/index")
def list_gt_docs():
    db = get_db()
    gt_coll = db[GT_COLL]
    stores = sorted({doc["store_id"] for doc in gt_coll.find({}, {"store_id": 1}) if doc.get("store_id")})
    days = sorted({doc["day_id"] for doc in gt_coll.find({}, {"day_id": 1}) if doc.get("day_id")})
    docs = [
        {"store_id": doc.get("store_id"), "day_id": doc.get("day_id")}
        for doc in gt_coll.find({}, {"store_id": 1, "day_id": 1})
        if doc.get("store_id") and doc.get("day_id")
    ]
    return {"stores": stores, "days": days, "docs": docs}


@app.get("/api/editor-state")
def get_editor_state():
    state = read_editor_state()
    return {"ok": state is not None, "state": state}


@app.post("/api/editor-state")
def save_editor_state(payload: Dict[str, Any] = Body(...)):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="state payload must be a JSON object")
    if not payload.get("people"):
        raise HTTPException(status_code=400, detail="state payload must include 'people'")

    write_editor_state(payload)

    people = payload.get("people") or []
    store_id = payload.get("store_id") or payload.get("store")
    day_id = payload.get("day_id") or payload.get("day")
    adjudicated = bool(payload.get("adjudicated", False))
    all_entries: List[Dict[str, Any]] = []
    persons_map: Dict[str, Any] = {}

    for person in people:
        pid = person.get("person_id")
        entries = []
        for ev in person.get("events", []):
            ev = dict(ev)
            ev.setdefault("store_id", store_id)
            ev.setdefault("day_id", day_id)
            all_entries.append(ev)
            entries.append(
                {
                    "entry_id": ev.get("entry_id"),
                    "alert_id": ev.get("alert_id"),
                    "store_id": ev.get("store_id"),
                    "day_id": ev.get("day_id"),
                    "camera": ev.get("camera"),
                    "timestamp": ev.get("timestamp"),
                    "direction": ev.get("direction"),
                    "images": ev.get("image_paths") or ev.get("images") or [],
                }
            )
        persons_map[pid] = {
            "alias": person.get("alias") or "",
            "first_seen": person.get("first_seen") or "",
            "entries": entries,
            "include": not bool(person.get("exclude", False)),
        }

    if all_entries:
        seen_ids = upsert_entries(all_entries)
        # remove stale entries for this store/day
        get_db()[ENTRIES_COLL].delete_many({"store_id": store_id, "day_id": day_id, "_id": {"$nin": seen_ids}})
    if persons_map:
        upsert_cluster(store_id, day_id, persons_map, adjudicated=adjudicated)

    return {"ok": True, "path": str(STATE_PATH)}
