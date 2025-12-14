
import argparse
import os
from pymongo import MongoClient

def fix_gt_clusters_camera_names(mongo_uri: str, db_name: str, collection_name: str, start_date: str, limit: int):
    """
    Corrects the 'camera' field within the nested 'persons.entries' array in the
    gt_tools.clusters collection by extracting the full camera name from the 'entry_id'.
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

    processed_doc_count = 0
    total_updated_entries = 0

    for doc in docs_to_process:
        processed_doc_count += 1
        cluster_id = doc.get("_id")
        persons = doc.get("persons", {})
        is_cluster_modified = False
        
        print(f"--- Processing cluster {processed_doc_count}/{doc_count}: {cluster_id} ---")

        for person_id, person_data in persons.items():
            entries = person_data.get("entries", [])
            for entry in entries:
                entry_id = entry.get("entry_id")
                if not entry_id:
                    continue

                parts = entry_id.split('-')
                if len(parts) > 2:
                    correct_camera_name = "-".join(parts[:-2])
                else:
                    print(f"  Skipping entry {entry_id}: ID has too few parts.")
                    continue

                current_camera_name = entry.get("camera")

                if current_camera_name != correct_camera_name:
                    print(f"  Updating entry {entry_id}: '{current_camera_name}' -> '{correct_camera_name}'")
                    entry["camera"] = correct_camera_name
                    is_cluster_modified = True
                    total_updated_entries += 1

        if is_cluster_modified:
            print(f"  Saving changes to cluster {cluster_id}")
            collection.update_one(
                {"_id": cluster_id},
                {"$set": {"persons": persons}}
            )
        else:
            print(f"  No changes needed for cluster {cluster_id}")

    print(f"\nFinished. Processed {processed_doc_count} clusters and updated a total of {total_updated_entries} nested entries.")

if __name__ == "__main__":
    default_mongo_uri = os.getenv("REID_MONGODB_URI", "mongodb://teknoir:change-me@localhost:37017")

    parser = argparse.ArgumentParser(
        description="Correct the 'camera' field in the gt_tools.clusters collection."
    )
    parser.add_argument(
        "--mongo-uri",
        default=default_mongo_uri,
        help=f"MongoDB connection string. Defaults to REID_MONGODB_URI env var or: {default_mongo_uri}"
    )
    parser.add_argument("--db", default="gt_tools", help="Database name.")
    parser.add_argument("--collection", default="clusters", help="Collection name.")
    parser.add_argument("--start-date", default="2025-12-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of documents to process (0 for all). Default is 0.")
    
    args = parser.parse_args()

    fix_gt_clusters_camera_names(
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        collection_name=args.collection,
        start_date=args.start_date,
        limit=args.limit
    )
