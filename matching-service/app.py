# app.py
import os
import logging

logging.basicConfig(level=logging.INFO)
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from matcher import ReEntryMatcher

# BE SURE TO MAKE SURE THIS MATCHES THE MODEL_CKPT IN THE run_local.sh file
MODEL_CKPT = os.getenv("MODEL_CKPT", "matching-service/models/encoder/model.pt")
ATTR_SCHEMA = os.getenv("ATTR_SCHEMA", "attr_schema.json")
BUCKET_PREFIX = os.getenv("BUCKET_PREFIX", "gs://victra-poc.teknoir.cloud")
MONGO_URI    = os.getenv("REID_MONGODB_URI", "mongodb://teknoir:change-me@localhost:37017")
MONGO_DB     = os.getenv("MONGO_DB", "reid_service")
MONGO_ENTRIES_COLLECTION = os.getenv("MONGO_ENTRIES_COLLECTION", "observations")
MONGO_EVENTS_COLLECTION = os.getenv("MONGO_EVENTS_COLLECTION", "line-crossings")


FUSION_MODE = "xattn"
# FUSION_MODE = "baseline"
MARGIN    = float(os.getenv("MARGIN", "0.00"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.86"))
TOPK      = int(os.getenv("TOPK", "20"))

app = FastAPI(title="Re-entry Matching Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = ReEntryMatcher(
    model_ckpt_path=MODEL_CKPT,
    attr_schema_path=ATTR_SCHEMA if os.path.exists(ATTR_SCHEMA) else None,
    mongo_uri=MONGO_URI if MONGO_URI else None,
    mongo_db=MONGO_DB,
    entries_collection=MONGO_ENTRIES_COLLECTION,
    events_collection=MONGO_EVENTS_COLLECTION,
    threshold=THRESHOLD,
    topk=TOPK,
)

class MatchRequest(BaseModel):
    day_id: str = Field(..., description="e.g., 2025-11-06")
    store_id: str = Field(..., description="store identifier (e.g., nc0009)")
    entry_id: str = Field(..., description="unique id for this line crossing")
    employee_id: Optional[str] = Field(default=None, description="employee match id, if any")
    alert_id: Optional[str] = Field(default=None, description="alert document id")
    timestamp: Optional[str] = Field(default=None, description="event timestamp")
    direction: Optional[str] = Field(default=None, description="entry/exit direction")
    camera: Optional[str] = Field(default=None, description="camera name")
    images: Optional[List[str]] = Field(default=None, description="source image URIs")
    embeddings: List[List[float]] = Field(..., description="list of per-image reid vectors (L2-normalized preferred)")
    attrs: Optional[Dict[str, Any]] = Field(default=None, description="normalized attribute dict per schema")
    topk: Optional[int] = Field(default=None)

class RebuildRequest(BaseModel):
    day_id: str

@app.get("/health")
def health():
    return {"ok": True, "threshold": matcher.threshold, "topk": matcher.topk}

@app.post("/match")
def match(req: MatchRequest):
    if not req.embeddings:
        raise HTTPException(status_code=400, detail="embeddings list is empty")
    out = matcher.query_then_add(
        day_id=req.day_id,
        entry_id=req.entry_id,
        store_id=req.store_id,
        employee_id=req.employee_id,
        alert_id=req.alert_id,
        timestamp=req.timestamp,
        direction=req.direction,
        camera=req.camera,
        image_uris=req.images,
        embeddings=req.embeddings,
        attrs=req.attrs or {},
        topk=req.topk,
    )
    # runtime rule is applied inside query_then_add (single-threshold cosine match)
    return {
        "ok": True,
        "images": req.images or [],
        "embeddings": req.embeddings,
        "timestamp": req.timestamp,
        "direction": req.direction,
        "camera": req.camera,
        **out,
    }

@app.post("/rebuild")
def rebuild(req: RebuildRequest):
    out = matcher.rebuild_from_mongo(req.day_id)
    if not out.get("ok"):
        raise HTTPException(status_code=400, detail=out.get("error", "unknown error"))
    return out


@app.get("/manifest")
def manifest(day_id: str, store_id: str, entry_id: Optional[str] = None, camera: Optional[List[str]] = None, emp_id: Optional[str] = None):
    try:
        if emp_id:
            data = matcher.build_employee_manifest(day_id, store_id, emp_id)
        else:
            data = matcher.build_manifest(day_id, store_id, entry_id, camera)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if BUCKET_PREFIX:
        for person in data.get("people", []):
            for event in person.get("events", []):
                images = event.get("images") or []
                event["images"] = [
                    img if not img or img.startswith("gs://") else f"{BUCKET_PREFIX.rstrip('/')}/{img.lstrip('/')}"
                    for img in images
                ]
    return {"ok": True, **data}


@app.get("/compare")
def compare(entry_a: str, entry_b: str):
    try:
        data = matcher.compare_entries(entry_a, entry_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, **data}
