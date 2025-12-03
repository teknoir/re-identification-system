
import argparse
import json
from datetime import datetime, timedelta

import requests
from pymongo import MongoClient


def post_matches_from_mongo(
    day_id: str,
    store_id: str,
    mongo_uri: str,
    mongo_db: str,
    entries_collection: str,
    match_url: str,
):
    """
    Pulls entries from mongo and posts them to the /match endpoint.
    """
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    entries = db[entries_collection]

    # The user-provided day is in EDT, but the database stores timestamps in UTC.
    # EDT is UTC-4, so we need to offset the start time by +4 hours.
    edt_day_start = datetime.strptime(day_id, "%Y-%m-%d")
    utc_offset = timedelta(hours=4)

    utc_start = edt_day_start + utc_offset
    utc_end = utc_start + timedelta(days=1)

    start_day_str = utc_start.strftime("%Y-%m-%dT%H:%M:%S")
    end_day_str = utc_end.strftime("%Y-%m-%dT%H:%M:%S")

    query = {
        "metadata.timestamp": {
            "$gte": start_day_str,
            "$lt": end_day_str,
        },
        "data.peripheral.name": {"$regex": f"^{store_id}"},
    }

    print(f"Query: {json.dumps(query, default=str)}")

    count = entries.count_documents(query)
    print(f"Found {count} entries to process.")

    for entry in entries.find(query):
        payload = {
            "day_id": day_id,
            "store_id": store_id,
            "entry_id": entry["data"]["id"],
            "alert_id": str(entry["_id"]),
            "timestamp": entry.get("metadata", {}).get("timestamp"),
            "direction": entry.get("metadata", {}).get("annotations", {}).get("teknoir.org/linedir"),
            "camera": entry.get("data", {}).get("peripheral", {}).get("name"),
            "images": entry["data"]["files"],
            "embeddings": entry["data"]["embeddings"],
            "attrs": entry["data"]["attributes"],
        }

        print(f"Posting entry {entry['data']['id']} to {match_url}")
        try:
            response = requests.post(match_url, json=payload)
            response.raise_for_status()
            print(f"Got response: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"Error posting entry {entry['data']['id']}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True, help="Day in YYYY-MM-DD format")
    parser.add_argument("--store", required=True, help="Store ID (e.g., nc0009)")
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://teknoir:change-me@localhost:37017/historian?authSource=admin",
    )
    parser.add_argument("--mongo-db", default="historian")
    # parser.add_argument("--entries-collection", default="entries")
    parser.add_argument("--match-url", default="http://0.0.0.0:8884/match")

    args = parser.parse_args()

    post_matches_from_mongo(
        day_id=args.day,
        store_id=args.store,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        entries_collection="line-crossings",
        match_url=args.match_url,
    )
