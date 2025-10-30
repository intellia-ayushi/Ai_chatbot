import os
from flask import request
from supabase import create_client, Client
from typing import Optional, Tuple


def get_supabase_client() -> Client:
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    if not url or not key:
        raise RuntimeError('Supabase credentials missing: SUPABASE_URL and key are required')
    return create_client(url, key)


def extract_bearer_token(req: request) -> Optional[str]:
    auth_header = req.headers.get('Authorization') or ''
    if auth_header.startswith('Bearer '):
        return auth_header.split(' ', 1)[1].strip()
    # Fallback from cookie if used by frontend
    return req.cookies.get('sb-access-token')


def verify_user(req: request) -> Tuple[Optional[str], Optional[dict]]:
    token = extract_bearer_token(req)
    if not token:
        return None, {'error': 'Missing bearer token'}
    supabase = get_supabase_client()
    try:
        user = supabase.auth.get_user(token)
        if not user or not getattr(user, 'user', None):
            return None, {'error': 'Invalid user'}
        return user.user.id, None
    except Exception as e:
        return None, {'error': str(e)}


def get_user_email(req: request) -> Optional[str]:
    token = extract_bearer_token(req)
    if not token:
        return None
    supabase = get_supabase_client()
    try:
        user = supabase.auth.get_user(token)
        return getattr(getattr(user, 'user', None), 'email', None)
    except Exception:
        return None


