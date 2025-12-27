"""CRUD helpers for analysis setup configuration."""

import json
from typing import Any, Dict, List

import streamlit as st
from sqlalchemy import text

from analysis.settings.config import get_engine
from analysis.settings.logging import log_exception


def _normalize_json(value: Any) -> Any:
    """Normalize JSON-like values from bytes/strings into Python objects."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, str):
                trimmed = parsed.strip()
                if trimmed.startswith("{") or trimmed.startswith("["):
                    return json.loads(trimmed)
            return parsed
        except json.JSONDecodeError:
            return value
    return value


def fetch_all_setups() -> List[Dict[str, Any]]:
    """
    Read all saved setup records.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT id, setup_name, description "
                    "FROM analysis_list_setups "
                    "ORDER BY setup_name"
                )
            )
            rows = [dict(row) for row in result.mappings()]
        return rows
    except Exception:
        log_exception("setup_repo.fetch_all_setups failed")
        return []


def fetch_all_setups_detail() -> List[Dict[str, Any]]:
    """
    Read setup list and parse note field from config_calculation.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT id, setup_name, description, config_calculation "
                    "FROM analysis_list_setups "
                    "ORDER BY setup_name"
                )
            )
            rows = [dict(row) for row in result.mappings()]

        for row in rows:
            calc_raw = _normalize_json(row.get("config_calculation"))
            note = ""
            if isinstance(calc_raw, dict):
                note = calc_raw.get("note", "") or ""
            row["note"] = note
            row.pop("config_calculation", None)

        return rows
    except Exception:
        log_exception("setup_repo.fetch_all_setups_detail failed")
        return []


def fetch_setup_config(setup_name: str) -> Dict[str, Any] | None:
    """
    Load extraction and calculation config for a setup.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT config_extraction, config_calculation "
                    "FROM analysis_list_setups "
                    "WHERE setup_name = :name"
                ),
                {"name": setup_name},
            ).mappings().first()

        if row is None:
            return None

        extraction = _normalize_json(row["config_extraction"])
        raw_calculation = _normalize_json(row["config_calculation"])

        if raw_calculation is None:
            calculation: Dict[str, Any] = {
                "calc_rules": [],
                "note": "",
                "exclusions": [],
                "pivot": {
                    "index": [],
                    "columns": [],
                    "values": [],
                    "agg": "mean",
                },
            }
        elif isinstance(raw_calculation, dict):
            calculation = dict(raw_calculation)
            calculation.setdefault("calc_rules", [])
            calculation["note"] = raw_calculation.get("note", "")
            calculation.setdefault("exclusions", [])
            calculation.setdefault(
                "pivot",
                {"index": [], "columns": [], "values": [], "agg": "mean"},
            )
        else:
            calculation = {
                "calc_rules": [],
                "note": "",
                "exclusions": [],
                "pivot": {"index": [], "columns": [], "values": [], "agg": "mean"},
            }

        return {"extraction": extraction, "calculation": calculation}
    except Exception as e:
        st.error(f"无法加载配置 `{setup_name}`: {e}")
        log_exception("setup_repo.fetch_setup_config failed", {"setup_name": setup_name})
        return None


def save_setup_config(setup_name: str, description: str | None, config: Dict[str, Any]) -> None:
    """Back-compat wrapper for save_extraction_config."""
    save_extraction_config(setup_name, description, config)


def save_extraction_config(
    setup_name: str, description: str | None, config_data: Dict[str, Any]
) -> None:
    """
    Save extraction config to analysis_list_setups.config_extraction.
    """
    config_json = json.dumps(config_data, ensure_ascii=False)
    engine = get_engine()
    sql = text(
        "INSERT INTO analysis_list_setups (setup_name, description, config_extraction) "
        "VALUES (:name, :desc, :config) "
        "ON DUPLICATE KEY UPDATE "
        "description = VALUES(description), "
        "config_extraction = VALUES(config_extraction), "
        "updated_at = NOW()"
    )
    try:
        with engine.connect() as conn:
            conn.execute(
                sql,
                {
                    "name": setup_name,
                    "desc": description or "",
                    "config": config_json,
                },
            )
            conn.commit()
    except Exception as e:
        st.error(f"保存一段配置 `{setup_name}` 失败: {e}")
        log_exception("setup_repo.save_extraction_config failed", {"setup_name": setup_name})
    finally:
        engine.dispose()


def save_calculation_config(setup_name: str, config_data: Dict[str, Any]) -> None:
    """
    Save calculation config to analysis_list_setups.config_calculation.
    """
    config_json = json.dumps(config_data, ensure_ascii=False)
    engine = get_engine()
    sql = text(
        "UPDATE analysis_list_setups "
        "SET config_calculation = :config_calculation, "
        "    updated_at = NOW() "
        "WHERE setup_name = :name"
    )
    try:
        with engine.connect() as conn:
            conn.execute(
                sql,
                {"name": setup_name, "config_calculation": config_json},
            )
            conn.commit()
    except Exception as e:
        st.error(f"保存二段配置 `{setup_name}` 失败: {e}")
        log_exception("setup_repo.save_calculation_config failed", {"setup_name": setup_name})
    finally:
        engine.dispose()


def delete_setup_config(setup_name: str) -> None:
    """
    Delete a setup by name.
    """
    engine = get_engine()
    sql = text("DELETE FROM analysis_list_setups WHERE setup_name = :name")
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"name": setup_name})
            conn.commit()
    except Exception as e:
        st.error(f"删除配置 `{setup_name}` 失败: {e}")
        log_exception("setup_repo.delete_setup_config failed", {"setup_name": setup_name})
    finally:
        engine.dispose()
