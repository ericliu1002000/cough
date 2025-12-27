"""Logging helpers for access and error events."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from analysis.auth import session as auth_session

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
ACCESS_LOG_PATH = LOG_DIR / "access.log"
ERROR_LOG_PATH = LOG_DIR / "error.log"


def _ensure_log_dir() -> None:
    """Ensure the log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _format_fields(fields: Dict[str, Any]) -> str:
    """Format extra fields into a space-separated key=value string."""
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _configure_logger(name: str, path: Path, level: int) -> logging.Logger:
    """Configure and return a logger with a rotating file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    _ensure_log_dir()
    logger.setLevel(level)
    logger.propagate = False

    handler = RotatingFileHandler(
        path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)

    return logger


def get_access_logger() -> logging.Logger:
    """Return the access logger instance."""
    return _configure_logger("analysis.access", ACCESS_LOG_PATH, logging.INFO)


def get_error_logger() -> logging.Logger:
    """Return the error logger instance."""
    return _configure_logger("analysis.error", ERROR_LOG_PATH, logging.ERROR)


def log_access(
    page: str,
    user: str | None = None,
    details: Dict[str, Any] | None = None,
    dedupe: bool = True,
) -> None:
    """Write a page access entry, optionally deduped per session."""
    if not page:
        return

    if dedupe:
        key = f"access_logged::{page}"
        if st.session_state.get(key):
            return
        st.session_state[key] = True

    if user is None:
        user = st.session_state.get(auth_session.SESSION_USER_KEY, "-")

    fields = {"page": page, "user": user}
    if details:
        fields.update(details)

    message = "access " + _format_fields(fields)
    get_access_logger().info(message)


def log_event(action: str, user: str | None = None, details: Dict[str, Any] | None = None) -> None:
    """Write a generic access event entry."""
    fields = {"action": action}
    if user is not None:
        fields["user"] = user
    if details:
        fields.update(details)
    message = "event " + _format_fields(fields)
    get_access_logger().info(message)


def log_error(message: str, details: Dict[str, Any] | None = None) -> None:
    """Write an error message without an exception stack."""
    fields = dict(details or {})
    suffix = _format_fields(fields)
    text = f"{message} {suffix}".strip()
    get_error_logger().error(text)


def log_exception(message: str, details: Dict[str, Any] | None = None) -> None:
    """Write an error message with the current exception stack."""
    fields = dict(details or {})
    suffix = _format_fields(fields)
    text = f"{message} {suffix}".strip()
    get_error_logger().exception(text)
