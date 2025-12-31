"""Shared page-level formatting utilities."""

from urllib.parse import urlencode

import streamlit as st
from streamlit import config as st_config

from analysis.auth.session import append_auth_token


def build_page_url(page_name: str, params: dict[str, str] | None = None) -> str:
    """Build a page URL with the configured base path and query params."""
    base_path = st_config.get_option("server.baseUrlPath") or ""
    base_prefix = f"/{base_path.strip('/')}" if base_path else ""
    params = append_auth_token(params)
    if params:
        query = urlencode(params)
        return f"{base_prefix}/{page_name}?{query}"
    return f"{base_prefix}/{page_name}"


def truncate_text(text: str, max_len: int) -> str:
    """Truncate text with ellipsis when it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def hide_login_sidebar_entry() -> None:
    """Hide the login page entry in the Streamlit sidebar navigation."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] a[href*="_login"],
        section[data-testid="stSidebar"] a[href*="login"],
        section[data-testid="stSidebar"] a[aria-label*="Login"],
        section[data-testid="stSidebar"] button[aria-label*="Login"],
        section[data-testid="stSidebar"] li:has(a[href*="_login"]),
        section[data-testid="stSidebar"] li:has(a[href*="login"]),
        section[data-testid="stSidebar"] li:has(a[aria-label*="Login"]),
        section[data-testid="stSidebar"] li:has(button[aria-label*="Login"]) {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
