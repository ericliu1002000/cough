"""Session state helpers for the data builder page."""

import streamlit as st


def init_filter_rows() -> None:
    """Ensure filter row state exists."""
    if "filter_rows" not in st.session_state:
        st.session_state.filter_rows = []


def add_filter_row() -> None:
    """Append a blank filter row placeholder."""
    init_filter_rows()
    st.session_state.filter_rows.append({"id": len(st.session_state.filter_rows)})


def remove_filter_row(idx: int) -> None:
    """Remove a filter row by index."""
    init_filter_rows()
    if 0 <= idx < len(st.session_state.filter_rows):
        st.session_state.filter_rows.pop(idx)
