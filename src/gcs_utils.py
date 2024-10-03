# gcs_utils.py
import os
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

def initialize_storage_client():
    """
    Initializes and returns a Google Cloud Storage client.
    """
    try:
        client = storage.Client()
        return client
    except Exception as e:
        print(f"Error initializing Google Cloud Storage client: {e}")
        raise e

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a blob from the bucket.
    """
    client = initialize_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    try:
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}.")
    except GoogleAPIError as e:
        print(f"Error downloading blob: {e}")
        raise e

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket.
    """
    client = initialize_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    try:
        blob.upload_from_filename(source_file_name)
        print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}.")
    except GoogleAPIError as e:
        print(f"Error uploading blob: {e}")
        raise e

def list_blobs(bucket_name, prefix=None):
    """
    Lists all blobs in the bucket with an optional prefix.
    """
    client = initialize_storage_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return blobs
