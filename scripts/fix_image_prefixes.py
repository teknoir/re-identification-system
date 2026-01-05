#!/usr/bin/env python3
"""
Replace a URL prefix in gt_tools image/file paths for clusters and entries.

Default scope is the gt_tools.clusters and gt_tools.entries collections for all
days/stores. You can narrow with --store-id / --day-id, and run with --dry-run
to inspect without writing.
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Tuple

from pymongo import MongoClient

OLD_PREFIX = "http://localhost:8882/victra-poc/media-service/api/jpeg/media"
NEW_PREFIX = "gs://victra-poc.teknoir.cloud/media"


def replace_prefix(paths: Iterable[str], old: str, new: str) -> Tuple[List[str], int]:
    """Return updated list and number of replacements."""
    out: List[str] = []
    changed = 0
    for p in paths:
        if isinstance(p, str) and p.startswith(old):
            out.append(new + p[len(old) :])
            changed += 1
        else:
            out.append(p)
    return out, changed


def process_clusters(coll, query, old, new, dry_run: bool) -> Tuple[int, List[str]]:
    modified = 0
    details: List[str] = []
    for doc in coll.find(query):
        persons = doc.get("persons") or {}
        doc_changed = False
        touched_pids: List[str] = []
        for pid, person in persons.items():
            entries = person.get("entries") or []
            for entry in entries:
                images = entry.get("images") or []
                new_images, cnt = replace_prefix(images, old, new)
                if cnt:
                    entry["images"] = new_images
                    doc_changed = True
                    if pid not in touched_pids:
                        touched_pids.append(pid)
        if doc_changed:
            modified += 1
            details.append(f"{doc.get('store_id')}-{doc.get('day_id')} persons: {', '.join(sorted(touched_pids))}")
            if not dry_run:
                coll.replace_one({"_id": doc.get("_id")}, doc)
    return modified, details


def process_entries(coll, query, old, new, dry_run: bool) -> Tuple[int, List[str]]:
    modified = 0
    details: List[str] = []
    for doc in coll.find(query, {"files": 1, "store_id": 1, "day_id": 1}):
        files = doc.get("files") or []
        new_files, cnt = replace_prefix(files, old, new)
        if cnt:
            modified += 1
            details.append(f"{doc.get('_id')} ({doc.get('store_id')}-{doc.get('day_id')})")
            if not dry_run:
                coll.update_one({"_id": doc.get("_id")}, {"$set": {"files": new_files}})
    return modified, details


def main():
    parser = argparse.ArgumentParser(description="Replace image/file URL prefixes in gt_tools.")
    parser.add_argument("--mongo-uri", default=os.getenv("MANIFEST_EDITOR_MONGO", "mongodb://teknoir:change-me@localhost:37017"))
    parser.add_argument("--db", default=os.getenv("MANIFEST_EDITOR_DB", "gt_tools"))
    parser.add_argument("--clusters-coll", default=os.getenv("MANIFEST_EDITOR_CLUSTERS_COLL", "clusters"))
    parser.add_argument("--entries-coll", default=os.getenv("MANIFEST_EDITOR_ENTRIES_COLL", "entries"))
    parser.add_argument("--old-prefix", default=OLD_PREFIX)
    parser.add_argument("--new-prefix", default=NEW_PREFIX)
    parser.add_argument("--store-id", help="Limit to a specific store_id")
    parser.add_argument("--day-id", help="Limit to a specific day_id")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    query = {}
    if args.store_id:
        query["store_id"] = args.store_id
    if args.day_id:
        query["day_id"] = args.day_id

    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    clusters_coll = db[args.clusters_coll]
    entries_coll = db[args.entries_coll]

    clusters_changed, clusters_detail = process_clusters(clusters_coll, query, args.old_prefix, args.new_prefix, args.dry_run)
    entries_changed, entries_detail = process_entries(entries_coll, query, args.old_prefix, args.new_prefix, args.dry_run)

    print(
        f"Done. Clusters touched: {clusters_changed}, Entries touched: {entries_changed}. "
        f"{'DRY RUN (no writes performed)' if args.dry_run else 'Writes applied.'}"
    )
    if clusters_detail:
        print("Clusters updated:")
        for line in clusters_detail:
            print(f"  - {line}")
    if entries_detail:
        print("Entries updated:")
        for line in entries_detail:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
