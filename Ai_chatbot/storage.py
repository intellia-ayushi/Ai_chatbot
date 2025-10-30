import os
import boto3
from botocore.config import Config
try:
    import b2sdk.v2 as b2
except Exception:
    b2 = None

# Temporary hardcoded Backblaze B2 configuration for debugging
# Set FORCE_B2 to True to ignore .env and always use these values
FORCE_B2 = True
HARDCODED_B2_ENDPOINT = "https://s3.us-west-002.backblazeb2.com"
HARDCODED_B2_PUBLIC = "https://s3.us-west-002.backblazeb2.com"
HARDCODED_B2_KEY_ID = "005ceb9303bb4310000000013"
HARDCODED_B2_APP_KEY = "K005WibKn9W6UbetVyO/ds4sbQJyzBU"
HARDCODED_B2_BUCKET = "vitebite"


def _env(name: str, *aliases: str, default: str = "") -> str:
    for key in (name, *aliases):
        val = os.getenv(key)
        if val:
            return val
    return default


def get_b2_client():
    """Create an S3-compatible client for Backblaze B2 using env vars.

    Supported env names (any one of each group works):
    - Endpoint: B2_ENDPOINT, B2_S3_ENDPOINT
    - Key ID:   B2_KEY_ID, B2_APPLICATION_KEY_ID
    - App Key:  B2_APP_KEY, B2_APPLICATION_KEY
    """
    if FORCE_B2:
        endpoint = HARDCODED_B2_ENDPOINT
        key_id = HARDCODED_B2_KEY_ID
        app_key = HARDCODED_B2_APP_KEY
        print('[B2] Using HARDCODED config')
    else:
        # Correct mapping from env
        endpoint = _env('B2_ENDPOINT', 'B2_S3_ENDPOINT')
        key_id = _env('B2_KEY_ID', 'B2_APPLICATION_KEY_ID')
        app_key = _env('B2_APP_KEY', 'B2_APPLICATION_KEY')
    if not (endpoint and key_id and app_key):
        print('[B2] Missing configuration. endpoint/key_id/app_key required')
        print('[B2] endpoint=', endpoint)
        print('[B2] key_id set? ', bool(key_id))
        return None
    try:
        print('[B2] Creating client with endpoint:', endpoint)
        s3cfg = Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        return boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            region_name='us-west-002',
            config=s3cfg,
        )
    except Exception as e:
        print('[B2] Failed to create client:', e)
        return None


def get_bucket_name() -> str:
    if FORCE_B2:
        return HARDCODED_B2_BUCKET
    return _env('B2_BUCKET', 'B2_BUCKET_NAME')


def get_public_base() -> str:
    # Prefer explicit public url; else fall back to endpoint
    if FORCE_B2:
        return HARDCODED_B2_PUBLIC
    return _env('B2_PUBLIC_URL', 'B2_ENDPOINT', 'B2_S3_ENDPOINT')


def upload_file_to_b2(local_path: str, object_key: str) -> str | None:
    """Upload a file to Backblaze B2 using native b2sdk only and return public URL."""
    if not b2:
        print('[B2] b2sdk is not installed')
        return None
    try:
        # Resolve creds and bucket
        if FORCE_B2:
            key_id = HARDCODED_B2_KEY_ID
            app_key = HARDCODED_B2_APP_KEY
            bucket_name = HARDCODED_B2_BUCKET
        else:
            key_id = _env('B2_KEY_ID', 'B2_APPLICATION_KEY_ID')
            app_key = _env('B2_APP_KEY', 'B2_APPLICATION_KEY')
            bucket_name = _env('B2_BUCKET', 'B2_BUCKET_NAME')

        print('[B2] Using b2sdk upload')
        info = b2.InMemoryAccountInfo()
        b2_api = b2.B2Api(info)
        b2_api.authorize_account('production', key_id, app_key)
        bucket_obj = b2_api.get_bucket_by_name(bucket_name)

        print(f"[B2] Uploading {local_path} -> b2://{bucket_name}/{object_key}")
        bucket_obj.upload_local_file(local_file=local_path, file_name=object_key)

        # Build public download URL (works for public buckets)
        url = b2_api.get_download_url_for_file_name(bucket_name, object_key)
        print('[B2] Uploaded URL:', url)
        return url
    except Exception as e:
        print('[B2] b2sdk upload failed:', e)
        return None


