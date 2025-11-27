"""Google Cloud Storage helpers."""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Optional, Tuple

from google.cloud import storage
from google.oauth2 import service_account

from .config import get_settings, Settings


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"URI must start with gs://, got {uri}")
    bucket_and_path = uri[5:]
    parts = bucket_and_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return parts[0], parts[1]


@lru_cache(maxsize=1)
def _get_storage_client() -> storage.Client:
    settings = get_settings()
    if settings.google_credentials_file:
        credentials = service_account.Credentials.from_service_account_file(
            settings.google_credentials_file
        )
        return storage.Client(credentials=credentials)
    return storage.Client()


def build_public_url(uri: str, settings: Settings) -> str:
    bucket, key = parse_gcs_uri(uri)
    # return f"https://storage.googleapis.com/{bucket}/{key}"
    return f"https://{settings.domain}/{settings.namespace}/media-service/api/jpeg/{key}"

def get_image_url(uri: str) -> str:
    settings = get_settings()
    if not uri.startswith("gs://"):
        return uri
    if not settings.gcs_url_signed:
        return build_public_url(uri, settings)

    bucket_name, blob_name = parse_gcs_uri(uri)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    expiration = dt.timedelta(seconds=settings.gcs_url_ttl_seconds)
    return blob.generate_signed_url(expiration=expiration, version="v4")


def download_blob_bytes(uri: str) -> bytes:
    """Download a GCS object into memory and return raw bytes."""
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {uri}")
    return blob.download_as_bytes()
