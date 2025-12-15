#!/usr/bin/env python3
"""Export encoder training manifests from adjudicated MongoDB clusters.

The script reads adjudicated clusters and their entries and emits files
compatible with the encoder training pipeline (see model/encoder/train_encoder.sh):

- a multicluster GT file mapping person_ids to their entries
- an entry manifest mapping entry_ids to metadata (images/embeddings/attrs)

Only clusters with adjudicated=True are used, and only persons with include=True
inside those clusters are included. Defaults mirror the manifest editor server
but can be overridden via flags or environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Set

from pymongo import MongoClient
from pymongo.collection import Collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MANIFEST_EDITOR_MONGO", "mongodb://teknoir:change-me@localhost:37017"),
        help="Mongo connection string",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("MANIFEST_EDITOR_DB", "gt_tools"),
        help="Database name",
    )
    parser.add_argument(
        "--clusters-coll",
        default=os.getenv("MANIFEST_EDITOR_CLUSTERS_COLL", "clusters"),
        help="Clusters collection name",
    )
    parser.add_argument(
        "--entries-coll",
        default=os.getenv("MANIFEST_EDITOR_ENTRIES_COLL", "entries"),
        help="Entries collection name",
    )
    parser.add_argument(
        "--gt-coll",
        default=os.getenv("MANIFEST_EDITOR_GT_COLL", "map"),
        help="Ground truth collection name.",
    )
    parser.add_argument(
        "--store-id",
        action="append",
        help="Limit to specific store_id (repeatable).",
    )
    parser.add_argument(
        "--day-id",
        action="append",
        help="Limit to specific day_id (repeatable).",
    )
    parser.add_argument(
        "--allow-unadjudicated",
        action="store_true",
        help="Include clusters even if adjudicated flag is false/missing.",
    )
    parser.add_argument(
        "--no-day-suffix",
        dest="add_day_suffix",
        action="store_false",
        help="Do not append the day suffix to person_ids.",
    )
    parser.set_defaults(add_day_suffix=True)
    parser.add_argument(
        "--entry-output",
        type=Path,
        default=Path("model/encoder/runs/entry.json"),
        help="Output path for the entry manifest.",
    )
    parser.add_argument(
        "--gt-output",
        type=Path,
        default=Path("model/encoder/runs/multi_gt.json"),
        help="Output path for the multicluster GT file.",
    )
    parser.add_argument(
        "--drop-gt",
        action="store_true",
        help="Drop the ground truth collection before exporting.",
    )
    return parser.parse_args()


def ensure_day_suffix(person_id: str, day_id: Optional[str], enabled: bool) -> str:
    if not enabled or not day_id or not isinstance(day_id, str):
        return person_id
    suffix = day_id[5:10] if len(day_id) >= 10 and day_id[4] == "-" else day_id
    tail = f"-{suffix}"
    return person_id if person_id.endswith(tail) else f"{person_id}{tail}"


def merge_images(*candidates: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        for img in candidate:
            if not img or not isinstance(img, str) or img in seen:
                continue
            seen.add(img)
            out.append(img)
    return out


def merge_entry_payload(
    entry_ref: Mapping[str, Any],
    entry_doc: Mapping[str, Any],
    default_store: Optional[str],
    default_day: Optional[str],
) -> Dict[str, Any]:
    def pick(*vals):
        for v in vals:
            if v not in (None, ""):
                return v
        return None

    images = merge_images(
        entry_ref.get("image_paths"),
        entry_ref.get("images"),
        entry_doc.get("files"),
        entry_doc.get("images"),
    )
    embeddings = entry_doc.get("embeddings") or []
    attrs = entry_ref.get("attrs") or entry_doc.get("attrs") or {}

    payload: Dict[str, Any] = {
        "store_id": pick(entry_ref.get("store_id"), entry_doc.get("store_id"), default_store),
        "day_id": pick(entry_ref.get("day_id"), entry_doc.get("day_id"), default_day),
        "camera": pick(entry_ref.get("camera"), entry_doc.get("camera")),
        "timestamp": pick(entry_ref.get("timestamp"), entry_doc.get("timestamp")),
        "direction": pick(entry_ref.get("direction"), entry_doc.get("direction")),
        "alert_id": pick(entry_ref.get("alert_id"), entry_doc.get("alert_id")),
        "images": images,
    }
    if embeddings:
        payload["embeddings"] = embeddings
    if attrs:
        payload["attrs"] = attrs
    return payload


def merge_entry_manifest(existing: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> None:
    # Always merge images to get a full list
    existing["images"] = merge_images(existing.get("images"), incoming.get("images"))

    # Update other metadata, preferring incoming values only if existing is empty
    for key in ("store_id", "day_id", "camera", "timestamp", "direction", "alert_id", "attrs"):
        if key not in existing or existing.get(key) in (None, "", [], {}):
            if incoming.get(key):
                existing[key] = incoming[key]

    # If incoming payload has non-empty embeddings, always use them
    if incoming.get("embeddings"):
        existing["embeddings"] = incoming["embeddings"]


def load_existing_entry_includes(gt_coll: Collection, clusters: Iterable[Mapping[str, Any]]) -> Dict[str, bool]:
    """Fetch include flags from the GT map collection for the clusters we are about to export."""
    keys: Set[str] = set()
    for cluster in clusters:
        store = cluster.get("store_id")
        day = cluster.get("day_id")
        if store and day:
            keys.add(f"{store}-{day}")
    if not keys:
        return {}
    includes: Dict[str, bool] = {}
    cursor = gt_coll.find({"_id": {"$in": list(keys)}}, {"entry_map": 1})
    for doc in cursor:
        entry_map = doc.get("entry_map") or {}
        for entry_id, meta in entry_map.items():
            include = meta.get("include")
            if include is not None:
                includes[entry_id] = bool(include)
    return includes


def main() -> None:
    args = parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    clusters_coll = db[args.clusters_coll]
    entries_coll = db[args.entries_coll]
    gt_coll = db[args.gt_coll]

    cluster_filter: Dict[str, Any] = {}
    if not args.allow_unadjudicated:
        cluster_filter["adjudicated"] = True
    if args.store_id:
        cluster_filter["store_id"] = {"$in": args.store_id}
    if args.day_id:
        cluster_filter["day_id"] = {"$in": args.day_id}

    clusters = list(clusters_coll.find(cluster_filter))
    if not clusters:
        raise SystemExit("No clusters matched the provided filters.")

    existing_entry_includes = load_existing_entry_includes(gt_coll, clusters)

    if args.drop_gt:
        print(f"Dropping ground truth collection: {args.db}.{args.gt_coll}")
        gt_coll.drop()

    persons_raw: Dict[str, Dict[str, Any]] = {}
    clusters_data: List[Dict[str, Any]] = []
    all_entry_ids: Set[str] = set()

    for cluster in clusters:
        store_id = cluster.get("store_id")
        day_id = cluster.get("day_id")
        persons = cluster.get("persons") or {}
        cluster_persons: Dict[str, Dict[str, Any]] = {}
        for raw_pid, person in persons.items():
            pid = ensure_day_suffix(str(raw_pid), day_id, enabled=args.add_day_suffix)
            target = persons_raw.setdefault(
                pid,
                {"alias": person.get("alias") or "", "first_seen": person.get("first_seen") or "", "entries_raw": []},
            )
            if not target.get("alias") and person.get("alias"):
                target["alias"] = person["alias"]
            if not target.get("first_seen") and person.get("first_seen"):
                target["first_seen"] = person["first_seen"]
            for entry_ref in person.get("entries") or []:
                entry_id = entry_ref.get("entry_id")
                if not entry_id:
                    continue
                all_entry_ids.add(entry_id)
                target["entries_raw"].append(
                    {"entry": entry_ref, "store_id": store_id, "day_id": day_id, "entry_id": entry_id}
                )
                cluster_target = cluster_persons.setdefault(
                    pid,
                    {
                        "alias": person.get("alias") or "",
                        "first_seen": person.get("first_seen") or "",
                        "source_id": raw_pid,
                        "include": bool(person.get("include", True)),
                        "entry_ids": [],
                    },
                )
                cluster_target["entry_ids"].append(entry_id)
        clusters_data.append(
            {
                "store_id": store_id,
                "day_id": day_id,
                "persons": cluster_persons,
            }
        )

    singletons = 0
    multi_entry_clusters = 0
    for person in persons_raw.values():
        if len(person.get("entries_raw", [])) == 1:
            singletons += 1
        else:
            multi_entry_clusters += 1
    print("\n--- Cluster Stats ---")
    print(f"Singleton clusters: {singletons}")
    print(f"Multi-entry clusters: {multi_entry_clusters}")
    print(f"Found {len(clusters)} matching clusters containing {len(all_entry_ids)} unique entry IDs.")

    entries_docs: Dict[str, Dict[str, Any]] = {}
    if all_entry_ids:
        print(f"Fetching {len(all_entry_ids)} full entry documents from DB...")
        entries_cursor = entries_coll.find({"_id": {"$in": list(all_entry_ids)}})
        for doc in entries_cursor:
            entries_docs[doc.get("_id")] = doc
        with_embeddings = sum(1 for doc in entries_docs.values() if doc.get("embeddings"))
        print(f"-> Fetched {len(entries_docs)} documents, {with_embeddings} have non-empty embeddings.")

    entry_manifest: Dict[str, Dict[str, Any]] = {}
    entry_person_map_all: Dict[str, str] = {}
    filtered_assignments: Dict[str, str] = {}

    for pid, pdata in persons_raw.items():
        enriched_entries: List[Dict[str, Any]] = []
        for ref in pdata.get("entries_raw", []):
            entry_id = ref["entry_id"]
            entry_doc = entries_docs.get(entry_id, {})
            payload = merge_entry_payload(ref["entry"], entry_doc, ref.get("store_id"), ref.get("day_id"))
            merge_entry_manifest(entry_manifest.setdefault(entry_id, {}), payload)
            if entry_id:
                entry_person_map_all[entry_id] = pid

    # Persist per-store/day GT docs in Mongo
    for cluster_doc in clusters_data:
        store_id = cluster_doc.get("store_id")
        day_id = cluster_doc.get("day_id")
        key = f"{store_id}-{day_id}"
        persons_meta: Dict[str, Dict[str, Any]] = {}
        entry_map: Dict[str, Dict[str, Any]] = {}
        for pid, pdata in cluster_doc.get("persons", {}).items():
            entry_ids = pdata.get("entry_ids") or []
            persons_meta[pid] = {
                "alias": pdata.get("alias") or "",
                "first_seen": pdata.get("first_seen") or "",
            }
            for entry_id in entry_ids:
                # Always prefer the include flag from the person cluster data.
                include_flag = bool(pdata.get("include", True))
                entry_map[entry_id] = {"person_id": pid, "include": include_flag}
                existing_entry_includes[entry_id] = include_flag
                if include_flag:
                    filtered_assignments[entry_id] = pid
        doc = {
            "_id": key,
            "store_id": store_id,
            "day_id": day_id,
            "entry_map": entry_map,
            "persons": persons_meta,
            "meta": {"exported_at": datetime.now(timezone.utc).isoformat()},
        }
        gt_coll.replace_one({"_id": key}, doc, upsert=True)

    # Write combined files
    args.entry_output.parent.mkdir(parents=True, exist_ok=True)
    args.gt_output.parent.mkdir(parents=True, exist_ok=True)

    total_assignments = len(entry_person_map_all)
    # Filter entry_manifest to only include entries present in filtered_map (strict mode)
    strict_entry_manifest = {eid: data for eid, data in entry_manifest.items() if eid in filtered_assignments}

    final_with_embeddings = sum(1 for data in strict_entry_manifest.values() if data.get("embeddings"))

    args.entry_output.write_text(json.dumps(strict_entry_manifest, indent=2))

    args.gt_output.write_text(json.dumps(filtered_assignments, indent=2))

    print("\n--- Output ---")
    print(
        f"Wrote {len(strict_entry_manifest)} entries ({final_with_embeddings} with embeddings) to {args.entry_output}"
    )
    print(
        f"Wrote {len(filtered_assignments)} assignments (from {total_assignments} total) to {args.gt_output}"
    )
    print(f"Updated {len(clusters_data)} GT docs in Mongo")


if __name__ == "__main__":
    main()
