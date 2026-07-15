import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_supabase_client = None
_configured = False

def get_supabase_client():
    global _supabase_client, _configured
    if _configured:
        return _supabase_client

    _configured = True
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        logger.info("Supabase credentials not found; remote sync disabled.")
        return None

    try:
        from supabase import create_client, Client
        _supabase_client = create_client(url, key)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        
    return _supabase_client

def pull_from_supabase(local_path: str, remote_key: str, bucket_name: str = None):
    """
    Downloads a file from the Supabase Storage bucket to local_path if it exists remotely.
    Does nothing if the remote key doesn't exist yet (e.g. first run / empty bucket).
    """
    if bucket_name is None:
        bucket_name = os.environ.get("SUPABASE_BUCKET")
        
    client = get_supabase_client()
    if not client or not bucket_name:
        return
        
    try:
        # Download the file from Supabase storage
        response = client.storage.from_(bucket_name).download(remote_key)
        
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Write the binary data to the local file
        with open(local_path, "wb") as f:
            f.write(response)
            
        logger.info(f"Successfully pulled {remote_key} from Supabase to {local_path}.")
    except Exception as e:
        # StorageException is thrown if the file doesn't exist or on network issues.
        logger.warning(f"Failed to pull {remote_key} from Supabase. Falling back to local data. Reason: {e}")

def push_to_supabase(local_path: str, remote_key: str, bucket_name: str = None):
    """
    Uploads a local file to the bucket, overwriting the remote copy using upsert.
    """
    if bucket_name is None:
        bucket_name = os.environ.get("SUPABASE_BUCKET")
        
    client = get_supabase_client()
    if not client or not bucket_name:
        return
        
    if not os.path.exists(local_path):
        logger.warning(f"Local file {local_path} does not exist; skipping push.")
        return
        
    try:
        with open(local_path, "rb") as f:
            client.storage.from_(bucket_name).upload(
                file=f,
                path=remote_key,
                file_options={"upsert": "true"}
            )
        logger.info(f"Successfully pushed {local_path} to Supabase at {remote_key}.")
    except Exception as e:
        logger.error(f"Failed to push {local_path} to Supabase. Reason: {e}")
