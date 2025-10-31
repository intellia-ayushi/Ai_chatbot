import os
from typing import List, Optional, Dict, Any
from supabase import create_client, Client


def get_supabase_client() -> Client:
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    if not url or not key:
        raise RuntimeError('Supabase credentials missing: SUPABASE_URL and key are required')
    return create_client(url, key)


def ensure_tables():
    # This is a no-op placeholder. Use SQL migrations in Supabase dashboard.
    # Expected tables:
    # documents(id uuid pk, user_id uuid, path text, filename text, created_at timestamptz)
    # chats(id uuid pk, user_id uuid, title text, created_at timestamptz)
    # messages(id uuid pk, chat_id uuid, role text, content text, created_at timestamptz)
    pass


def add_document(jwt: str, user_id: str, path: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Add a document for a user: Save path+filename as a new entry in the user's documents array.
    If this is the first upload for the user, create a new row. Requires the user's JWT for RLS insert!
    """
    sb = get_supabase_client()
    try:
        sb.postgrest.auth(jwt)  # CRUCIAL: use user's JWT for RLS-upsert!
        resp = sb.table('documents').select('documents').eq('user_id', user_id).execute()
        docs = []
        if resp.data and len(resp.data) > 0 and isinstance(resp.data[0].get('documents'), list):
            docs = resp.data[0]['documents']
        docs.append({'path': path, 'filename': filename})
        upsert_resp = sb.table('documents').upsert({'user_id': user_id, 'documents': docs}, on_conflict='user_id').execute()
        print(f'[DEBUG][add_document] user_id={user_id}, documents array now: {docs}')
        return (upsert_resp.data or [None])[0]
    except Exception as e:
        print(f'[ERROR][add_document] Failed for user_id={user_id}: {e}')
        return None

def list_user_documents(jwt: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all document info for the user as a list (from the single array row); pass jwt for RLS protection.
    """
    sb = get_supabase_client()
    try:
        sb.postgrest.auth(jwt)
        resp = sb.table('documents').select('documents').eq('user_id', user_id).execute()
        if resp.data and len(resp.data) > 0 and isinstance(resp.data[0].get('documents'), list):
            print(f'[DEBUG][list_user_documents] user_id={user_id}, documents: {resp.data[0]["documents"]}')
            return resp.data[0]['documents']
    except Exception as e:
        print(f'[ERROR][list_user_documents] Could not fetch documents for user_id={user_id}: {e}')
    return []


def create_chat(user_id: str, title: str) -> Optional[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = sb.table('chats').insert({'user_id': user_id, 'title': title}).execute()
    return (resp.data or [None])[0]


def list_user_chats(user_id: str) -> List[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = sb.table('chats').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
    return resp.data or []


def add_message(chat_id: str, role: str, content: str) -> Optional[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = sb.table('messages').insert({'chat_id': chat_id, 'role': role, 'content': content}).execute()
    return (resp.data or [None])[0]


def list_messages(chat_id: str) -> List[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = sb.table('messages').select('*').eq('chat_id', chat_id).order('created_at', asc=True).execute()
    return resp.data or []


def upsert_profile(user_id: str, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Create or update a simple profiles row for this user. Table schema:
    profiles(id uuid primary key, email text, created_at timestamptz default now())
    """
    sb = get_supabase_client()
    data = {'id': user_id}
    if email:
        data['email'] = email
    try:
        resp = sb.table('profiles').upsert(data, on_conflict='id').execute()
        return (resp.data or [None])[0]
    except Exception:
        # Profiles table might not exist; safe to ignore
        return None


def list_profiles() -> List[Dict[str, Any]]:
    """List all user profiles (requires service-role key or permissive RLS)."""
    sb = get_supabase_client()
    try:
        resp = sb.table('profiles').select('*').order('created_at', desc=True).execute()
        return resp.data or []
    except Exception:
        return []


def upsert_profile_as_user(jwt: str, user_id: str, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Upsert profiles row using the user's JWT so RLS policies pass without service role.
    Requires the 'profiles owner' policy (auth.uid() = id).
    """
    if not jwt:
        return upsert_profile(user_id, email)
    sb = get_supabase_client()
    try:
        # act as the user for this request
        sb.postgrest.auth(jwt)
        data = {'id': user_id}
        if email:
            data['email'] = email
        resp = sb.table('profiles').upsert(data, on_conflict='id').execute()
        # clear override
        sb.postgrest.auth(None)
        return (resp.data or [None])[0]
    except Exception:
        try:
            sb.postgrest.auth(None)
        except Exception:
            pass
        return None


# --- New: per-user message history stored as a JSON array ---

def get_user_messages(user_id: str) -> List[Dict[str, Any]]:
    """Return messages array for a user from user_messages table. If row missing, return []."""
    sb = get_supabase_client()
    try:
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).single().execute()
        if resp.data and isinstance(resp.data.get('messages'), list):
            return resp.data['messages']
    except Exception:
        pass
    return []


from typing import List, Dict, Any

def get_user_messages_as_user(jwt: str, user_id: str) -> List[Dict[str, Any]]:
    """Fetch user messages while authenticating with JWT."""
    print("🔹 get_user_messages_as_user called")
    print(f"JWT provided: {'Yes' if jwt else 'No'}")
    print(f"User ID: {user_id}")

    if not jwt:
        print("⚠️ No JWT provided, using fallback get_user_messages()")
        return get_user_messages(user_id)

    sb = get_supabase_client()
    try:
        print("🌀 Authenticating with user JWT...")
        sb.postgrest.auth(jwt)

        print("🌀 Fetching messages from Supabase...")
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).single().execute()
        print("📦 Raw response:", resp)

        if resp.data:
            messages = resp.data.get('messages')
            if isinstance(messages, list):
                print(f"✅ Retrieved {len(messages)} messages")
                return messages
            else:
                print("⚠️ 'messages' is not a list:", type(messages))
        else:
            print("⚠️ No data found for user_id:", user_id)

    except Exception as e:
        print("❌ Error while fetching user messages:", e)

    finally:
        # Reset authentication cleanly here
        try:
            sb.postgrest.auth(None)
            print("🔁 Auth reset successfully")
        except Exception as e:
            print("⚠️ Error resetting auth:", e)

    print("⚠️ Returning empty list due to error")
    return []


def append_user_message(user_id: str, role: str, content: str) -> None:
    """Append a message to the user's messages array (creates row if missing)."""
    sb = get_supabase_client()
    current = get_user_messages(user_id)
    current.append({ 'role': role, 'content': content })
    try:
        # upsert the aggregated array
        sb.table('user_messages').upsert({ 'user_id': user_id, 'messages': current }, on_conflict='user_id').execute()
    except Exception:
        pass


def append_user_message_as_user(jwt: str, user_id: str, role: str, content: str) -> None:
    """
    Append a message for a user.
    ✅ If user_id does not exist in user_messages table, automatically creates a new row.
    ✅ Works with or without JWT (handles RLS).
    """
    sb = get_supabase_client()
    try:
        if jwt:
            sb.postgrest.auth(jwt)

        # 🌀 Try to fetch existing messages (no .single() to avoid empty result error)
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).execute()

        if resp.data and len(resp.data) > 0 and isinstance(resp.data[0].get('messages'), list):
            msgs = resp.data[0]['messages']
            print(f"[DEBUG] Found existing messages for {user_id}: {len(msgs)}")
        else:
            # 🆕 No row found — create new one
            msgs = []
            print(f"[DEBUG] No existing messages for {user_id}, creating new row...")

        # Add new message
        msgs.append({'role': role, 'content': content})

        # 📝 Upsert (insert or update)
        sb.table('user_messages').upsert(
            {'user_id': user_id, 'messages': msgs},
            on_conflict='user_id'
        ).execute()

        print(f"[SUCCESS] Message added for user_id={user_id}, total={len(msgs)}")

    except Exception as e:
        print(f"[ERROR][append_user_message_as_user] Failed for user_id={user_id}: {e}")

    finally:
        # Reset auth to avoid affecting later queries
        try:
            sb.postgrest.auth(None)
        except Exception:
            pass



