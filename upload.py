import os
import sys
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

def upload_directory_to_gcs(local_directory, bucket_name, gcs_prefix='agi/'):
    """
    Uploads the contents of a local directory to a Google Cloud Storage bucket.

    :param local_directory: Path to the local directory to upload.
    :param bucket_name: Name of the GCS bucket.
    :param gcs_prefix: GCS prefix (folder path) within the bucket.
    """
    # Initialize the GCS client
    try:
        client = storage.Client()
    except Exception as e:
        print(f"Error initializing Google Cloud Storage client: {e}")
        sys.exit(1)

    # Get the bucket
    try:
        bucket = client.get_bucket(bucket_name)
    except GoogleAPIError as e:
        print(f"Error accessing bucket '{bucket_name}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error accessing bucket '{bucket_name}': {e}")
        sys.exit(1)

    # Walk through the local directory
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)

            # Compute the relative path to maintain directory structure in GCS
            relative_path = os.path.relpath(local_path, local_directory)
            gcs_path = os.path.join(gcs_prefix, relative_path).replace("\\", "/")  # For Windows compatibility

            # Create a blob object
            blob = bucket.blob(gcs_path)

            # Upload the file
            try:
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
            except FileNotFoundError:
                print(f"Error: The file {local_path} was not found.")
            except GoogleAPIError as e:
                print(f"Google API error uploading {local_path}: {e}")
            except Exception as e:
                print(f"Unexpected error uploading {local_path}: {e}")

    print("All files have been uploaded successfully.")

if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Upload a local directory to a Google Cloud Storage bucket.')
    parser.add_argument('--local-dir', type=str, default='data',
                        help='Path to the local directory to upload. Default is "data".')
    parser.add_argument('--bucket', type=str, required=True,
                        help='Name of the GCS bucket to upload to.')
    parser.add_argument('--prefix', type=str, default='agi/',
                        help='GCS prefix (folder path) within the bucket. Default is "agi/".')

    args = parser.parse_args()

    # Validate the local directory
    if not os.path.isdir(args.local_dir):
        print(f"Error: The directory '{args.local_dir}' does not exist.")
        sys.exit(1)

    # Call the upload function
    upload_directory_to_gcs(args.local_dir, args.bucket, args.prefix)
