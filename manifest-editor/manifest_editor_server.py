from __future__ import annotations

import json
import mimetypes
import os
from datetime import date, datetime, timedelta
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
STATUS_GRID_PATH = Path(__file__).with_name("processing_status.html")
GT_BROWSER_PATH = Path(__file__).with_name("ground_truth_browser.html")
ENCODER_STATIC_DIR = ROOT / "model" / "encoder" / "runs"
STATE_PATH = Path(os.getenv("MANIFEST_EDITOR_STATE", "/tmp/gt/manifest_editor_state.json"))
MANIFEST_API_BASE = os.getenv("MANIFEST_API_BASE", "http://matching-service")
BLOB_BASE = os.getenv("MANIFEST_EDITOR_BUCKET", "gs://victra-poc.teknoir.cloud")  # optional prefix for relative files
BASE_URL =  os.getenv("BASE_URL", "/")  # optional base url
MANIFEST_API_TIMEOUT_SECONDS = int(os.getenv("MANIFEST_API_TIMEOUT_SECONDS", "60"))
REID_MONGO_URI = os.getenv("REID_MONGODB_URI", "mongodb://teknoir:change-me@localhost:37017")
REID_MONGO_DB = "reid_service"
OBSERVATIONS_COLL = "observations"
MANIFEST_EDITOR_DB = os.getenv("MANIFEST_EDITOR_DB", "gt_tools")
ENTRIES_COLL = os.getenv("MANIFEST_EDITOR_ENTRIES_COLL", "entries")
CLUSTERS_COLL = os.getenv("MANIFEST_EDITOR_CLUSTERS_COLL", "clusters")
GT_COLL = os.getenv("MANIFEST_EDITOR_GT_COLL", "map")

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


def get_manifest_editor_db():
    global mongo_client
    if mongo_client is None:
        mongo_client = MongoClient(REID_MONGO_URI)
    return mongo_client[MANIFEST_EDITOR_DB]


reid_mongo_client: Optional[MongoClient] = None


def get_reid_db():
    global reid_mongo_client
    if reid_mongo_client is None:
        reid_mongo_client = MongoClient(REID_MONGO_URI)
    return reid_mongo_client[REID_MONGO_DB]


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
    db = get_manifest_editor_db()
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
    db = get_manifest_editor_db()
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
    db = get_manifest_editor_db()
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
    html = API_EDITOR_PATH.read_text(encoding="utf-8")
    # inject BASE_URL for client-side usage
    inject = f"<script>window.BASE_URL = {json.dumps(BASE_URL.rstrip('/'))};</script>"
    return inject + html

@app.get("/manifest_editor", response_class=HTMLResponse)
def manifest_api_editor():
    if not API_EDITOR_PATH.exists():
        raise HTTPException(status_code=500, detail="manifest_api_editor.html not found")
    html = API_EDITOR_PATH.read_text(encoding="utf-8")
    inject = f"<script>window.BASE_URL = {json.dumps(BASE_URL.rstrip('/'))};</script>"
    return inject + html


@app.get("/encoder_viewer", response_class=HTMLResponse)
def encoder_viewer():
    if not ENCODER_VIEWER_PATH.exists():
        raise HTTPException(status_code=500, detail="encoder_dataset_viewer.html not found")
    html = ENCODER_VIEWER_PATH.read_text(encoding="utf-8")
    inject = f"<script>window.BASE_URL = {json.dumps(BASE_URL.rstrip('/'))};</script>"
    return inject + html


@app.get("/gt_editor", response_class=HTMLResponse)
def gt_editor():
    if not GT_EDITOR_PATH.exists():
        raise HTTPException(status_code=500, detail="gt_editor.html not found")
    html = GT_EDITOR_PATH.read_text(encoding="utf-8")
    inject = f"<script>window.BASE_URL = {json.dumps(BASE_URL.rstrip('/'))};</script>"
    return inject + html


@app.get("/status_grid", response_class=HTMLResponse)
def status_grid_viewer():
    if not STATUS_GRID_PATH.exists():
        raise HTTPException(status_code=500, detail="processing_status.html not found")
    html = STATUS_GRID_PATH.read_text(encoding="utf-8")
    inject = f"<script>window.BASE_URL = {json.dumps(BASE_URL.rstrip('/'))};</script>"
    return inject + html


@app.get("/gt_browser", response_class=HTMLResponse)
def gt_browser_viewer():
    if not GT_BROWSER_PATH.exists():
        raise HTTPException(status_code=500, detail="ground_truth_browser.html not found")
    html = GT_BROWSER_PATH.read_text(encoding="utf-8")
    inject = f"<script>window.BASE_URL = {json.dumps(BASE_URL.rstrip('/'))};</script>"
    return inject + html

# Serve encoder run artifacts (large files) for viewer defaults
if ENCODER_STATIC_DIR.exists():
    app.mount("/static/encoder", StaticFiles(directory=str(ENCODER_STATIC_DIR)), name="encoder_static")

def _resolve_image_uri(source: str) -> str:
    if source.startswith(("gs://", "http://", "https://")):
        return source
    if BLOB_BASE:
        return f"{BLOB_BASE.rstrip('/')}/{source.lstrip('/')}"
    raise HTTPException(status_code=404, detail="Unable to resolve image path; set MANIFEST_EDITOR_BUCKET for relative paths")


@app.get("/api/grid-status")
def grid_status(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    # 1. Set date range
    if not end_date:
        end_date = date.today().isoformat()
    if not start_date:
        start_date = (date.today() - timedelta(days=30)).isoformat()

    date_match = {"day_id": {"$gte": start_date, "$lte": end_date}}

    # 2. Get DB and collections from gt_tools
    db = get_manifest_editor_db()
    clusters_coll = db[CLUSTERS_COLL]
    entries_coll = db[ENTRIES_COLL]

    # 3. Get all relevant clusters
    clusters = list(clusters_coll.find(date_match, {"day_id": 1, "store_id": 1, "adjudicated": 1}))
    
    if not clusters:
        return {"days": [], "stores": [], "grid_data": []}

    # 4. Get entry counts for the day/store pairs found in clusters
    day_store_pairs = list(set((c["day_id"], c["store_id"]) for c in clusters))
    match_query = {"$or": [{"day_id": d, "store_id": s} for d, s in day_store_pairs]}

    entry_count_pipeline = [
        {"$match": match_query},
        {"$group": {"_id": {"day_id": "$day_id", "store_id": "$store_id"}, "count": {"$sum": 1}}}
    ]
    entry_counts = {
        (item['_id']['day_id'], item['_id']['store_id']): item['count']
        for item in entries_coll.aggregate(entry_count_pipeline)
    }

    # 5. Build grid data and determine status
    grid_data = []
    day_set = set()
    store_set = set()

    for cluster in clusters:
        day_id = cluster["day_id"]
        store_id = cluster["store_id"]
        key = (day_id, store_id)
        
        day_set.add(day_id)
        store_set.add(store_id)
        
        count = entry_counts.get(key, 0)
        status = "none" # Default to Gray

        if cluster.get("adjudicated"):
            status = "adjudicated" # Green
        elif count > 0:
            status = "processed" # Yellow

        grid_data.append({
            "day_id": day_id,
            "store_id": store_id,
            "count": count,
            "status": status
        })

    all_days = sorted(list(day_set), reverse=True)
    all_stores = sorted(list(store_set))

    return {
        "days": all_days,
        "stores": all_stores,
        "grid_data": grid_data
    }

@app.get("/api/ground-truth-clusters")
def get_ground_truth_clusters(
    store_id: str, 
    day_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    db = get_manifest_editor_db()
    gt_coll = db[GT_COLL]
    entries_coll = db[ENTRIES_COLL]

    # 1. Get the entry -> person map
    key = f"{store_id}-{day_id}"
    map_doc = gt_coll.find_one({"_id": key}) or gt_coll.find_one({"store_id": store_id, "day_id": day_id})
    if not map_doc:
        raise HTTPException(status_code=404, detail="Ground truth map not found for store/day")

    entry_map = map_doc.get("entry_map", {})
    if not entry_map:
        return {"people": [], "total_people": 0, "page": 1, "page_size": page_size, "total_pages": 0}

    # 2. Group entry_ids by person_id, respecting the 'include' flag
    person_to_entries = {}
    for entry_id, mapping_info in entry_map.items():
        # Only include entries that are not explicitly excluded
        if not mapping_info.get("include", True):
            continue
        person_id = mapping_info.get("person_id")
        if person_id:
            person_to_entries.setdefault(person_id, []).append(entry_id)
    
    all_person_ids = sorted(person_to_entries.keys())
    total_people = len(all_person_ids)
    total_pages = (total_people + page_size - 1) // page_size

    # 3. Paginate the people
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_person_ids = all_person_ids[start_index:end_index]

    # 4. Fetch details only for entries of paginated people
    paginated_entry_ids = [eid for pid in paginated_person_ids for eid in person_to_entries[pid]]
    
    entry_details = {}
    if paginated_entry_ids:
        for doc in entries_coll.find({"_id": {"$in": paginated_entry_ids}}):
            entry_details[doc["_id"]] = doc

    # 5. Structure the final response
    people_out = []
    for person_id in paginated_person_ids:
        events_out = []
        for entry_id in person_to_entries[person_id]:
            details = entry_details.get(entry_id)
            if details:
                events_out.append({
                    "entry_id": entry_id,
                    "timestamp": details.get("timestamp"),
                    "direction": details.get("direction"),
                    "camera": details.get("camera"),
                    "images": details.get("files") or details.get("images") or details.get("image_paths", []),
                    "attrs": details.get("attrs", {})
                })
        
        people_out.append({
            "person_id": person_id,
            "events": events_out
        })

    return {
        "people": people_out,
        "total_people": total_people,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }

@app.get("/api/find-entry/{entry_id}")
def find_entry_in_gt_map(entry_id: str):
    db = get_manifest_editor_db()
    gt_coll = db[GT_COLL] # This is the 'map' collection

    # Find the document where the entry_id exists as a key in entry_map
    query = {f"entry_map.{entry_id}": {"$exists": True}}
    
    doc = gt_coll.find_one(query)

    if not doc:
        raise HTTPException(status_code=404, detail=f"Entry ID '{entry_id}' not found in any ground-truth map.")

    # Extract the person_id from the nested object
    person_id = doc.get("entry_map", {}).get(entry_id, {}).get("person_id")

    return {
        "store_id": doc.get("store_id"),
        "day_id": doc.get("day_id"),
        "person_id": person_id
    }


@app.get("/api/image")
def image_proxy(source: str = Query(..., description="gs:// URI or remote path to image")):
    uri = _resolve_image_uri(source)
    try:
        if uri.startswith("gs://"):
            data = download_blob_bytes(uri)
        else:
            resp = requests.get(uri, timeout=30)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch remote image: {uri}")
            data = resp.content
        mime = mimetypes.guess_type(uri)[0] or "image/jpeg"
        return StreamingResponse(iter([data]), media_type=mime)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors during image fetching
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching image: {str(e)}")


@app.get("/api/manifest-proxy")
def manifest_proxy(
    day_id: str,
    store_id: str,
    entry_id: Optional[str] = None,
    camera: Optional[List[str]] = Query(None),
    bypass_mongo: bool = Query(False, description="Bypass mongo check and go straight to the API"),
):
    # First try Mongo; fall back to API if not found
    if not bypass_mongo:
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
    db = get_manifest_editor_db()
    gt_coll = db[GT_COLL]
    entries_coll = db[ENTRIES_COLL]
    key = f"{store_id}-{day_id}"
    doc = gt_coll.find_one({"_id": key}) or gt_coll.find_one({"store_id": store_id, "day_id": day_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Ground truth not found for store/day")
    entry_map = doc.get("entry_map") or {}
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
    db = get_manifest_editor_db()
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
    db = get_manifest_editor_db()
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
        get_manifest_editor_db()[ENTRIES_COLL].delete_many({"store_id": store_id, "day_id": day_id, "_id": {"$nin": seen_ids}})
    if persons_map:
        upsert_cluster(store_id, day_id, persons_map, adjudicated=adjudicated)

    return {"ok": True, "path": str(STATE_PATH)}
