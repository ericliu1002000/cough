"""数据库配置读取服务。"""

from __future__ import annotations

import os
from typing import Dict, Optional


def get_business_db_config() -> Dict[str, str]:
    code = os.getenv("CURRENT_BUSINESS_CODE", "").strip()
    if not code:
        raise ValueError("未配置 CURRENT_BUSINESS_CODE。")

    prefix = f"{code.upper()}_"

    def _get_env(key: str) -> str:
        value = os.getenv(f"{prefix}{key}", "").strip()
        if not value:
            raise ValueError(f"缺少环境变量：{prefix}{key}")
        return value

    return {
        "code": code,
        "database": _get_env("MYSQL_DATABASE"),
        "user": _get_env("MYSQL_USER"),
        "password": _get_env("MYSQL_PASSWORD"),
        "host": _get_env("MYSQL_HOST"),
        "port": _get_env("MYSQL_PORT"),
    }


def get_business_db_name(config: Optional[Dict[str, str]] = None) -> str:
    if config is None:
        config = get_business_db_config()
    return config["database"]


def get_system_db_name() -> str:
    name = os.getenv("MYSQL_DATABASE", "").strip()
    if not name:
        raise ValueError("未配置 MYSQL_DATABASE。")
    return name


def get_system_engine():
    """系统库连接（包装 analysis.settings.config）。"""
    from analysis.settings.config import get_system_engine as _get_system_engine

    return _get_system_engine()


def get_business_engine():
    """业务库连接（包装 analysis.settings.config）。"""
    from analysis.settings.config import get_business_engine as _get_business_engine

    return _get_business_engine()
