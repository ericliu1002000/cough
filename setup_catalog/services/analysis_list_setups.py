"""analysis_list_setups 业务服务（系统库）。

负责系统库中的 analysis_list_setups 表的增删改查与配置解析。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st
from sqlalchemy import text

from analysis.settings.config import get_system_engine
from analysis.settings.logging import log_exception


def _normalize_json(value: Any) -> Any:
    """将数据库中读取的 JSON 字段规范化为 Python 对象。

    - 兼容 bytes/bytearray/str 三种常见存储格式
    - 允许字符串包裹的 JSON 再次解析
    - 解析失败时保留原值，避免误伤
    """
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
    """读取全部配置基础信息（用于列表展示）。"""
    try:
        # 只使用系统库，避免误读业务数据
        engine = get_system_engine()
        with engine.connect() as conn:
            # 列表页面只需要最小字段集合
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
        log_exception("analysis_list_setups.fetch_all_setups failed")
        return []


def fetch_all_setups_detail() -> List[Dict[str, Any]]:
    """读取配置列表，并从计算配置中提取 note 字段。"""
    try:
        engine = get_system_engine()
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
            # 兼容历史版本中 config_calculation 的不同存储格式
            calc_raw = _normalize_json(row.get("config_calculation"))
            note = ""
            if isinstance(calc_raw, dict):
                note = calc_raw.get("note", "") or ""
            row["note"] = note
            row.pop("config_calculation", None)

        return rows
    except Exception:
        log_exception("analysis_list_setups.fetch_all_setups_detail failed")
        return []


def fetch_setup_config(setup_name: str) -> Dict[str, Any] | None:
    """按名称加载配置（包含抽取与计算两段配置）。"""
    try:
        engine = get_system_engine()
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

        # 抽取配置保持原样返回，前端使用其完整结构
        extraction = _normalize_json(row["config_extraction"])
        raw_calculation = _normalize_json(row["config_calculation"])

        if raw_calculation is None:
            # 为空时提供标准默认结构，避免前端判断过多
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
            # 兜底填充缺失键，保证结构完整
            calculation = dict(raw_calculation)
            calculation.setdefault("calc_rules", [])
            calculation["note"] = raw_calculation.get("note", "")
            calculation.setdefault("exclusions", [])
            calculation.setdefault(
                "pivot",
                {"index": [], "columns": [], "values": [], "agg": "mean"},
            )
        else:
            # 非法格式时回退到默认结构
            calculation = {
                "calc_rules": [],
                "note": "",
                "exclusions": [],
                "pivot": {"index": [], "columns": [], "values": [], "agg": "mean"},
            }

        return {"extraction": extraction, "calculation": calculation}
    except Exception as exc:
        st.error(f"无法加载配置 `{setup_name}`: {exc}")
        log_exception(
            "analysis_list_setups.fetch_setup_config failed",
            {"setup_name": setup_name},
        )
        return None


def save_setup_config(
    setup_name: str, description: str | None, config: Dict[str, Any]
) -> None:
    """兼容入口：转发到保存一段配置。"""
    save_extraction_config(setup_name, description, config)


def save_extraction_config(
    setup_name: str, description: str | None, config_data: Dict[str, Any]
) -> None:
    """保存一段配置到 analysis_list_setups.config_extraction。"""
    # 使用 UTF-8 编码，保留中文字段或注释
    config_json = json.dumps(config_data, ensure_ascii=False)
    engine = get_system_engine()
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
            # 幂等写入：存在则更新，不存在则创建
            conn.execute(
                sql,
                {
                    "name": setup_name,
                    "desc": description or "",
                    "config": config_json,
                },
            )
            conn.commit()
    except Exception as exc:
        st.error(f"保存一段配置 `{setup_name}` 失败: {exc}")
        log_exception(
            "analysis_list_setups.save_extraction_config failed",
            {"setup_name": setup_name},
        )
    finally:
        engine.dispose()


def save_calculation_config(setup_name: str, config_data: Dict[str, Any]) -> None:
    """保存二段配置到 analysis_list_setups.config_calculation。"""
    config_json = json.dumps(config_data, ensure_ascii=False)
    engine = get_system_engine()
    sql = text(
        "UPDATE analysis_list_setups "
        "SET config_calculation = :config_calculation, "
        "    updated_at = NOW() "
        "WHERE setup_name = :name"
    )
    try:
        with engine.connect() as conn:
            # 只更新计算配置，避免覆盖抽取配置
            conn.execute(
                sql,
                {"name": setup_name, "config_calculation": config_json},
            )
            conn.commit()
    except Exception as exc:
        st.error(f"保存二段配置 `{setup_name}` 失败: {exc}")
        log_exception(
            "analysis_list_setups.save_calculation_config failed",
            {"setup_name": setup_name},
        )
    finally:
        engine.dispose()


def delete_setup_config(setup_name: str) -> None:
    """按名称删除配置。"""
    engine = get_system_engine()
    sql = text("DELETE FROM analysis_list_setups WHERE setup_name = :name")
    try:
        with engine.connect() as conn:
            # 删除后不可恢复，调用方需做好确认
            conn.execute(sql, {"name": setup_name})
            conn.commit()
    except Exception as exc:
        st.error(f"删除配置 `{setup_name}` 失败: {exc}")
        log_exception(
            "analysis_list_setups.delete_setup_config failed",
            {"setup_name": setup_name},
        )
    finally:
        engine.dispose()
