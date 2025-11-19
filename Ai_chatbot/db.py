import os
from typing import List, Optional, Dict, Any
from supabase import create_client, Client
from datetime import datetime


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


def add_document(jwt: str, user_id: str, path: str, filename: str, display_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Add a document for a user: Save path+filename as a new entry in the user's documents array.
    Optionally persist a human-friendly display_name (original filename uploaded by the user).
    Requires the user's JWT for RLS insert.
    """
    sb = get_supabase_client()
    try:
        sb.postgrest.auth(jwt)  # CRUCIAL: use user's JWT for RLS-upsert!
        resp = sb.table('documents').select('documents').eq('user_id', user_id).execute()
        docs = []
        if resp.data and len(resp.data) > 0 and isinstance(resp.data[0].get('documents'), list):
            docs = resp.data[0]['documents']
        doc_rec = {'path': path, 'filename': filename}
        if display_name:
            doc_rec['display_name'] = display_name
        docs.append(doc_rec)
        upsert_resp = sb.table('documents').upsert({'user_id': user_id, 'documents': docs}, on_conflict='user_id').execute()
        print(f'[DEBUG][add_document] user_id={user_id}, appended: {doc_rec}')
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


def get_last_answer_for_normalized_question(jwt: str, user_id: str, question_norm: str) -> Optional[str]:
    sb = get_supabase_client()
    try:
        if jwt:
            sb.postgrest.auth(jwt)
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).single().execute()
        msgs = (resp.data or {}).get('messages') if resp.data else None
        if not isinstance(msgs, list):
            return None
        last_answer = None
        for i in range(len(msgs)):
            item = msgs[i]
            if isinstance(item, dict) and item.get('role') == 'user':
                q = item.get('content')
                if isinstance(q, str):
                    norm = ' '.join(q.strip().lower().split())
                    if norm == question_norm:
                        if i + 1 < len(msgs):
                            nxt = msgs[i + 1]
                            if isinstance(nxt, dict) and nxt.get('role') == 'assistant':
                                ans = nxt.get('content')
                                if isinstance(ans, str) and ans:
                                    last_answer = ans
        return last_answer
    except Exception as e:
        print('[DB][last_answer][get] error:', e)
        return None
    finally:
        try:
            sb.postgrest.auth(None)
        except Exception:
            pass


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




# --- QA cache via user_messages JSON array ---

def get_cached_answer_from_messages(jwt: str, user_id: str, question_norm: str, ctx_hash: str) -> Optional[str]:
    """Scan user_messages.messages for a cached answer with matching question_norm and ctx_hash.
    Returns the latest matching answer or None.
    """
    sb = get_supabase_client()
    try:
        if jwt:
            sb.postgrest.auth(jwt)
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).single().execute()
        msgs = (resp.data or {}).get('messages') if resp.data else None
        if isinstance(msgs, list):
            # search latest first
            for item in reversed(msgs):
                if isinstance(item, dict) and item.get('type') == 'qa_cache':
                    if item.get('question_norm') == question_norm and item.get('ctx_hash') == ctx_hash:
                        ans = item.get('answer')
                        if isinstance(ans, str) and ans:
                            return ans
    except Exception as e:
        print('[DB][cache][get] error:', e)
    finally:
        try:
            sb.postgrest.auth(None)
        except Exception:
            pass
    return None


def upsert_cached_answer_in_messages(jwt: str, user_id: str, question_norm: str, ctx_hash: str, answer: str) -> None:
    """Insert or replace a cache entry in user_messages.messages for this (question_norm, ctx_hash).
    Keeps other messages untouched.
    """
    sb = get_supabase_client()
    try:
        if jwt:
            sb.postgrest.auth(jwt)
        # fetch current messages (no .single() to be tolerant of empty)
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).execute()
        msgs = []
        if resp.data and len(resp.data) > 0 and isinstance(resp.data[0].get('messages'), list):
            msgs = resp.data[0]['messages']
        # remove existing cache entry for same key
        filtered = []
        for item in (msgs or []):
            if isinstance(item, dict) and item.get('type') == 'qa_cache' and \
               item.get('question_norm') == question_norm and item.get('ctx_hash') == ctx_hash:
                continue
            filtered.append(item)
        # append new cache record
        filtered.append({
            'type': 'qa_cache',
            'question_norm': question_norm,
            'ctx_hash': ctx_hash,
            'answer': answer,
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        })
        sb.table('user_messages').upsert({'user_id': user_id, 'messages': filtered}, on_conflict='user_id').execute()
    except Exception as e:
        print('[DB][cache][upsert] error:', e)
    finally:
        try:
            sb.postgrest.auth(None)
        except Exception:
            pass


# --- Question-only cache (ignore context), first-answer wins ---
def get_qonly_cached_answer(jwt: str, user_id: str, question_norm: str) -> Optional[str]:
    sb = get_supabase_client()
    try:
        if jwt:
            sb.postgrest.auth(jwt)
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).single().execute()
        msgs = (resp.data or {}).get('messages') if resp.data else None
        if isinstance(msgs, list):
            for item in reversed(msgs):
                if isinstance(item, dict) and item.get('type') == 'qa_cache_qonly':
                    if item.get('question_norm') == question_norm:
                        ans = item.get('answer')
                        if isinstance(ans, str) and ans:
                            return ans
    except Exception as e:
        print('[DB][qonly][get] error:', e)
    finally:
        try:
            sb.postgrest.auth(None)
        except Exception:
            pass
    return None


def insert_qonly_cached_answer(jwt: str, user_id: str, question_norm: str, answer: str) -> None:
    sb = get_supabase_client()
    try:
        if jwt:
            sb.postgrest.auth(jwt)
        resp = sb.table('user_messages').select('messages').eq('user_id', user_id).execute()
        msgs = []
        if resp.data and len(resp.data) > 0 and isinstance(resp.data[0].get('messages'), list):
            msgs = resp.data[0]['messages']
        for item in msgs:
            if isinstance(item, dict) and item.get('type') == 'qa_cache_qonly' and item.get('question_norm') == question_norm:
                return
        msgs.append({
            'type': 'qa_cache_qonly',
            'question_norm': question_norm,
            'answer': answer,
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        })
        sb.table('user_messages').upsert({'user_id': user_id, 'messages': msgs}, on_conflict='user_id').execute()
    except Exception as e:
        print('[DB][qonly][insert] error:', e)
    finally:
        try:
            sb.postgrest.auth(None)
        except Exception:
            pass

