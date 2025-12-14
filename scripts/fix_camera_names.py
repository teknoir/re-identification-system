
import argparse
import os
from pymongo import MongoClient

def fix_camera_names(mongo_uri: str, db_name: str, collection_name: str, start_date: str, limit: int):
    """
    Corrects the 'camera' field in documents by extracting the full camera name
    from the 'entry_id' or '_id' field.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    query = {
        "day_id": {"$gte": start_date}
    }

    print(f"Connecting to MongoDB at {mongo_uri}...")
    print(f"Targeting database '{db_name}', collection '{collection_name}'.")
    print(f"Finding documents on or after day_id: {start_date}")

    cursor = collection.find(query)
    if limit > 0:
        cursor = cursor.limit(limit)
        print(f"Applying limit of {limit} documents.")

    docs_to_process = list(cursor)
    doc_count = len(docs_to_process)
    print(f"Found {doc_count} documents to process.")

    updated_count = 0
    skipped_count = 0
    processed_count = 0

    for i, doc in enumerate(docs_to_process):
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"--- Processed {processed_count}/{doc_count} documents ---")

        entry_id = doc.get("_id")
        if not entry_id or not isinstance(entry_id, str):
            print(f"Skipping document with missing or invalid _id: {doc}")
            skipped_count += 1
            continue

        parts = entry_id.split('-')
        if len(parts) > 2:
            correct_camera_name = "-".join(parts[:-2])
        else:
            print(f"Skipping doc {entry_id}: ID has too few parts.")
            skipped_count += 1
            continue

        current_camera_name = doc.get("camera")

        if current_camera_name != correct_camera_name:
            print(f"Updating {entry_id}: '{current_camera_name}' -> '{correct_camera_name}'")
            collection.update_one(
                {"_id": entry_id},
                {"$set": {"camera": correct_camera_name}}
            )
            updated_count += 1
        else:
            print(f"Skipping {entry_id}: already correct.")
            skipped_count += 1

    print(f"\nFinished processing {processed_count} documents.")
    print(f"Updated: {updated_count}")
    print(f"Skipped (already correct or invalid ID): {skipped_count}")

if __name__ == "__main__":
    # Set default connection string to the one used for reid_service
    default_mongo_uri = os.getenv("REID_MONGODB_URI", "mongodb://teknoir:change-me@localhost:37017")

    parser = argparse.ArgumentParser(
        description="Correct the 'camera' field in the observations collection by extracting it from the entry_id."
    )
    parser.add_argument(
        "--mongo-uri",
        default=default_mongo_uri,
        help=f"MongoDB connection string. Defaults to REID_MONGODB_URI env var or: {default_mongo_uri}"
    )
    parser.add_argument("--db", default="reid_service", help="Database name.")
    parser.add_argument("--collection", default="observations", help="Collection name.")
    parser.add_argument("--start-date", default="2025-12-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of documents to process (0 for all). Default is 0.")
    
    args = parser.parse_args()

    fix_camera_names(
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        collection_name=args.collection,
        start_date=args.start_date,
        limit=args.limit
    )
