"""Simple session-based authentication helpers."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv

SESSION_AUTH_KEY = "auth_authed"
SESSION_USER_KEY = "auth_user"
SESSION_LAST_ACTIVE_KEY = "auth_last_active"
SESSION_TOKEN_KEY = "auth_token"
NOTICE_KEY = "auth_notice"
AUTH_TOKEN_PARAM = "auth_token"
SESSION_TTL_SECONDS = 3 * 24 * 60 * 60

_BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_BASE_DIR / ".env")


@st.cache_resource
def _token_store() -> Dict[str, Dict[str, Any]]:
    """Return a global token store shared across sessions."""
    return {}


def get_credentials() -> tuple[str, str]:
    """Return expected username/password from environment variables."""
    user = os.getenv("APP_USERNAME", "admin")
    password = os.getenv("APP_PASSWORD", "admin")
    return user, password


def _get_query_param(name: str) -> str | None:
    """Return a query parameter value if present."""
    raw = st.query_params.get(name)
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    return str(raw)


def issue_token(username: str) -> str:
    """Create and store a new auth token for the user."""
    token = secrets.token_urlsafe(24)
    _token_store()[token] = {
        "user": username,
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    return token


def _get_token_payload(token: str) -> Dict[str, Any] | None:
    """Return token payload if valid and not expired."""
    store = _token_store()
    payload = store.get(token)
    if not payload:
        return None
    expires_at = payload.get("expires_at", 0)
    if time.time() > float(expires_at):
        store.pop(token, None)
        return None
    return payload


def refresh_token(token: str) -> Dict[str, Any] | None:
    """Extend token expiry if it is still valid."""
    payload = _get_token_payload(token)
    if not payload:
        return None
    payload["expires_at"] = time.time() + SESSION_TTL_SECONDS
    return payload


def invalidate_token(token: str) -> None:
    """Remove an auth token from the token store."""
    _token_store().pop(token, None)


def get_current_token() -> str | None:
    """Return the current auth token from session or query params."""
    token = st.session_state.get(SESSION_TOKEN_KEY)
    if token:
        return token
    return _get_query_param(AUTH_TOKEN_PARAM)


def append_auth_token(params: dict[str, str] | None) -> dict[str, str]:
    """Return params with auth token appended when available."""
    merged = dict(params or {})
    token = get_current_token()
    if token and AUTH_TOKEN_PARAM not in merged:
        merged[AUTH_TOKEN_PARAM] = token
    return merged


def authenticate(username: str, password: str) -> bool:
    """Return True when credentials match the configured values."""
    expected_user, expected_password = get_credentials()
    username = (username or "").strip()
    password = (password or "").strip()
    return username == expected_user and password == expected_password


def mark_authenticated(
    username: str, token: str | None = None, set_query: bool = True
) -> str:
    """Persist authentication state in session storage."""
    if token is None:
        token = issue_token(username)
    st.session_state[SESSION_AUTH_KEY] = True
    st.session_state[SESSION_USER_KEY] = username
    st.session_state[SESSION_LAST_ACTIVE_KEY] = time.time()
    st.session_state[SESSION_TOKEN_KEY] = token
    if set_query:
        st.query_params[AUTH_TOKEN_PARAM] = token
    return token


def sync_auth_from_query() -> bool:
    """Restore session auth from a valid auth token in query params."""
    token = _get_query_param(AUTH_TOKEN_PARAM)
    if not token:
        return False
    payload = _get_token_payload(token)
    if not payload:
        return False
    username = str(payload.get("user") or "")
    mark_authenticated(username, token=token, set_query=False)
    return True


def touch_session() -> None:
    """Update the last active timestamp for the current session."""
    st.session_state[SESSION_LAST_ACTIVE_KEY] = time.time()
    token = st.session_state.get(SESSION_TOKEN_KEY)
    if token:
        refresh_token(token)


def logout(clear_notice: bool = True) -> None:
    """Clear authentication state from session storage."""
    token = st.session_state.get(SESSION_TOKEN_KEY)
    if token:
        invalidate_token(token)
    for key in (
        SESSION_AUTH_KEY,
        SESSION_USER_KEY,
        SESSION_LAST_ACTIVE_KEY,
        SESSION_TOKEN_KEY,
    ):
        st.session_state.pop(key, None)
    if clear_notice:
        st.session_state.pop(NOTICE_KEY, None)


def set_notice(message: str) -> None:
    """Store a one-time notice to show on the login page."""
    st.session_state[NOTICE_KEY] = message


def pop_notice() -> str | None:
    """Return and clear the current notice message."""
    return st.session_state.pop(NOTICE_KEY, None)


def is_session_valid(ttl_seconds: int = SESSION_TTL_SECONDS) -> bool:
    """Return True when the session is authenticated and not expired."""
    if st.session_state.get(SESSION_AUTH_KEY) is not True:
        return False

    last_active = st.session_state.get(SESSION_LAST_ACTIVE_KEY)
    if not isinstance(last_active, (int, float)):
        return False

    if time.time() - float(last_active) > ttl_seconds:
        return False

    token = st.session_state.get(SESSION_TOKEN_KEY)
    if not token:
        return False
    if _get_token_payload(token) is None:
        return False

    return True


def require_login(redirect_page: str = "pages/_login.py") -> None:
    """Redirect to login when the session is missing or expired."""
    if is_session_valid():
        touch_session()
        return

    if sync_auth_from_query():
        touch_session()
        return

    if st.session_state.get(SESSION_AUTH_KEY):
        set_notice("登录已过期，请重新登录。")
    logout(clear_notice=False)
    st.switch_page(redirect_page)
    st.stop()
