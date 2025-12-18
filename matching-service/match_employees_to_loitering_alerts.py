#!/usr/bin/env python3
"""
Batch job script to match employees with employee loitering alerts.

This script:
1. Fetches all employee loitering alerts (type=employee_loitering) from historian.alerts
   that haven't been matched to an employee yet (no employee_id field exists).
2. For each alert, looks up the detection entry using the manifest-proxy API with entry_lookup.
3. Matches the entry_id with detection_id to find the employee_id.
4. Persists the employee_id to the alert (even if null, so the alert won't be reprocessed).
5. If employee_id is not null, sets llm_classification.is_valid to True.

Usage:
    python match_employees_to_loitering_alerts.py [options]

Environment variables:
    HISTORIAN_MONGODB_URI: MongoDB connection string for historian database
    MANIFEST_API_BASE: Base URL for the manifest-proxy API (default: http://localhost:8000)
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import requests
from pymongo import MongoClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_HISTORIAN_MONGO_URI = os.getenv(
    "HISTORIAN_MONGODB_URI", "mongodb://teknoir:change-me@localhost:27017"
)
DEFAULT_MANIFEST_API_BASE = os.getenv("MANIFEST_API_BASE", "http://localhost:8000")
DEFAULT_HISTORIAN_DB = "historian"
DEFAULT_ALERTS_COLLECTION = "alerts"


def lookup_entry_employee(
    manifest_api_base: str,
    detection_id: str,
    timeout: int = 60,
) -> Tuple[Optional[str], bool]:
    """
    Look up the employee_id for a detection entry via the manifest-proxy API.
    
    Args:
        manifest_api_base: Base URL of the manifest API
        detection_id: The detection/entry ID to look up
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (employee_id, entry_found):
        - employee_id: The employee ID if found, None otherwise
        - entry_found: True if the entry was found in the manifest, False otherwise
    """
    url = f"{manifest_api_base.rstrip('/')}/api/manifest-proxy"
    params = {
        "entry_lookup": detection_id,
    }
    
    try:
        logger.debug(f"Calling manifest API: {url} with params {params}")
        response = requests.get(url, params=params, timeout=timeout)
        
        if response.status_code != 200:
            logger.warning(
                f"Manifest API returned status {response.status_code} for "
                f"detection_id={detection_id}: {response.text}"
            )
            return None, False
        
        data = response.json()
        
        if not data.get("ok"):
            logger.warning(f"Manifest API returned ok=false: {data}")
            return None, False
        
        # Search through people and events for the matching entry_id
        people = data.get("people", [])
        for person in people:
            events = person.get("events", [])
            for event in events:
                entry_id = event.get("entry_id")
                if entry_id == detection_id:
                    # Found the matching entry
                    employee_id = event.get("employee_id")
                    logger.debug(
                        f"Found entry {detection_id} with employee_id={employee_id}"
                    )
                    return employee_id, True
        
        # Entry not found in manifest
        logger.debug(f"Entry {detection_id} not found in manifest response")
        return None, False
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling manifest API for detection_id={detection_id}")
        return None, False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling manifest API: {e}")
        return None, False
    except Exception as e:
        logger.error(f"Unexpected error looking up entry: {e}")
        return None, False


def update_alert_with_employee(
    alerts_collection,
    alert_id: str,
    employee_id: Optional[str],
    dry_run: bool = False,
) -> bool:
    """
    Update an alert with the retrieved employee_id.
    
    If employee_id is not null, also sets llm_classification.is_valid to True.
    
    Args:
        alerts_collection: MongoDB collection for alerts
        alert_id: The alert document _id
        employee_id: The employee ID to set (can be None)
        dry_run: If True, don't actually update the database
    
    Returns:
        True if the update was successful, False otherwise
    """
    try:
        update_doc: Dict[str, Any] = {
            "$set": {
                "employee_id": employee_id,
            }
        }
        
        # If employee_id is not null, also set llm_classification.is_valid to True
        if employee_id is not None:
            update_doc["$set"]["llm_classification.is_valid"] = True
        
        if dry_run:
            logger.info(f"[DRY RUN] Would update alert {alert_id} with employee_id={employee_id}")
            return True
        
        result = alerts_collection.update_one({"_id": alert_id}, update_doc)
        
        if result.modified_count > 0:
            logger.debug(f"Updated alert {alert_id} with employee_id={employee_id}")
            return True
        elif result.matched_count > 0:
            logger.debug(f"Alert {alert_id} matched but not modified (may be unchanged)")
            return True
        else:
            logger.warning(f"Alert {alert_id} not found for update")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update alert {alert_id}: {e}")
        return False


def process_employee_loitering_alerts(
    historian_mongo_uri: str,
    historian_db: str,
    alerts_collection_name: str,
    manifest_api_base: str,
    limit: int = 0,
    dry_run: bool = False,
    batch_size: int = 100,
    start_time_gt: Optional[str] = None,
    start_time_lt: Optional[str] = None,
) -> Dict[str, int]:
    """
    Main function to process employee loitering alerts.
    
    Args:
        historian_mongo_uri: MongoDB connection string
        historian_db: Database name
        alerts_collection_name: Collection name for alerts
        manifest_api_base: Base URL for manifest API
        limit: Maximum number of alerts to process (0 for all)
        dry_run: If True, don't actually update the database
        batch_size: Number of alerts to process before logging progress
        start_time_gt: Filter alerts with start_time > this value (ISO 8601 format)
        start_time_lt: Filter alerts with start_time < this value (ISO 8601 format)
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_found": 0,
        "processed": 0,
        "matched_with_employee": 0,
        "matched_without_employee": 0,
        "entry_not_found": 0,
        "errors": 0,
        "skipped_no_detection_id": 0,
    }
    
    logger.info(f"Connecting to MongoDB at {historian_mongo_uri}...")
    client = MongoClient(historian_mongo_uri)
    db = client[historian_db]
    alerts_collection = db[alerts_collection_name]
    
    # Query for employee loitering alerts without employee_id
    query: Dict[str, Any] = {
        "type": "employee_loitering",
        "employee_id": {"$exists": False},
    }
    
    # Add start_time filters if provided
    if start_time_gt or start_time_lt:
        start_time_filter: Dict[str, str] = {}
        if start_time_gt:
            start_time_filter["$gt"] = start_time_gt
        if start_time_lt:
            start_time_filter["$lt"] = start_time_lt
        query["start_time"] = start_time_filter
    
    logger.info("Querying alerts collection for employee_loitering alerts without employee_id...")
    if start_time_gt:
        logger.info(f"  Filtering start_time > {start_time_gt}")
    if start_time_lt:
        logger.info(f"  Filtering start_time < {start_time_lt}")
    
    cursor = alerts_collection.find(query)
    if limit > 0:
        cursor = cursor.limit(limit)
        logger.info(f"Limiting to {limit} alerts")
    
    alerts = list(cursor)
    stats["total_found"] = len(alerts)
    logger.info(f"Found {stats['total_found']} alerts to process")
    
    if stats["total_found"] == 0:
        logger.info("No alerts to process. Exiting.")
        return stats
    
    for i, alert in enumerate(alerts):
        alert_id = alert.get("_id")
        detection_id = alert.get("detection_id")
        
        # Log progress
        if (i + 1) % batch_size == 0:
            logger.info(
                f"Progress: {i + 1}/{stats['total_found']} alerts processed. "
                f"Matched: {stats['matched_with_employee']}, "
                f"No employee: {stats['matched_without_employee']}, "
                f"Not found: {stats['entry_not_found']}"
            )
        
        # Validate required fields
        if not detection_id:
            logger.warning(f"Alert {alert_id} has no detection_id, skipping")
            stats["skipped_no_detection_id"] += 1
            continue
        
        # Look up employee_id via API using entry_lookup
        employee_id, entry_found = lookup_entry_employee(
            manifest_api_base, detection_id
        )
        
        if entry_found:
            # Update the alert with employee_id (even if None)
            success = update_alert_with_employee(
                alerts_collection, alert_id, employee_id, dry_run
            )
            
            if success:
                stats["processed"] += 1
                if employee_id is not None:
                    stats["matched_with_employee"] += 1
                    logger.info(
                        f"Alert {alert_id}: Matched with employee_id={employee_id}"
                    )
                else:
                    stats["matched_without_employee"] += 1
                    logger.debug(
                        f"Alert {alert_id}: Entry found but no employee_id"
                    )
            else:
                stats["errors"] += 1
        else:
            # Entry not found in manifest - still update with employee_id=None
            success = update_alert_with_employee(
                alerts_collection, alert_id, None, dry_run
            )
            if success:
                stats["entry_not_found"] += 1
                logger.debug(f"Alert {alert_id}: Entry not found in manifest")
            else:
                stats["errors"] += 1
    
    client.close()
    
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Total alerts found: {stats['total_found']}")
    logger.info(f"Successfully processed: {stats['processed']}")
    logger.info(f"Matched with employee: {stats['matched_with_employee']}")
    logger.info(f"Matched without employee (null): {stats['matched_without_employee']}")
    logger.info(f"Entry not found in manifest: {stats['entry_not_found']}")
    logger.info(f"Skipped (no detection_id): {stats['skipped_no_detection_id']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Match employees with employee loitering alerts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--historian-mongo-uri",
        default=DEFAULT_HISTORIAN_MONGO_URI,
        help="MongoDB connection string for historian. Default: HISTORIAN_MONGODB_URI env var",
    )
    parser.add_argument(
        "--historian-db",
        default=DEFAULT_HISTORIAN_DB,
        help=f"Historian database name. Default: {DEFAULT_HISTORIAN_DB}",
    )
    parser.add_argument(
        "--alerts-collection",
        default=DEFAULT_ALERTS_COLLECTION,
        help=f"Alerts collection name. Default: {DEFAULT_ALERTS_COLLECTION}",
    )
    parser.add_argument(
        "--manifest-api-base",
        default=DEFAULT_MANIFEST_API_BASE,
        help=f"Base URL for manifest API. Default: {DEFAULT_MANIFEST_API_BASE}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of alerts to process (0 for all). Default: 0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually update the database, just log what would be done",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of alerts to process before logging progress. Default: 100",
    )
    parser.add_argument(
        "--start-time-gt",
        type=str,
        default=None,
        help="Filter alerts with start_time > this value (ISO 8601 format, e.g. 2025-12-17T00:00:00.000Z)",
    )
    parser.add_argument(
        "--start-time-lt",
        type=str,
        default=None,
        help="Filter alerts with start_time < this value (ISO 8601 format, e.g. 2025-12-18T00:00:00.000Z)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No database changes will be made")
        logger.info("=" * 60)
    
    try:
        stats = process_employee_loitering_alerts(
            historian_mongo_uri=args.historian_mongo_uri,
            historian_db=args.historian_db,
            alerts_collection_name=args.alerts_collection,
            manifest_api_base=args.manifest_api_base,
            limit=args.limit,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            start_time_gt=args.start_time_gt,
            start_time_lt=args.start_time_lt,
        )
        
        # Exit with error code if there were errors
        if stats["errors"] > 0:
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
