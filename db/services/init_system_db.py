"""系统数据库初始化（幂等）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import text

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analysis.settings.config import ensure_database_exists, get_system_engine

SYSTEM_TABLES: Dict[str, Dict[str, object]] = {
    "analysis_list_setups": {
        "columns": [
            ("id", "INT NOT NULL AUTO_INCREMENT PRIMARY KEY"),
            ("setup_name", "VARCHAR(100) NOT NULL"),
            ("description", "VARCHAR(255) NULL"),
            ("config_extraction", "JSON NOT NULL"),
            ("config_calculation", "JSON NULL"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            (
                "updated_at",
                "DATETIME DEFAULT CURRENT_TIMESTAMP "
                "ON UPDATE CURRENT_TIMESTAMP",
            ),
        ],
        "unique_indexes": {
            "uq_analysis_list_setups_setup_name": ["setup_name"]
        },
    },
    "business_objects": {
        "columns": [
            ("id", "INT NOT NULL AUTO_INCREMENT PRIMARY KEY"),
            ("project_name", "VARCHAR(100) NOT NULL"),
            ("object_name", "VARCHAR(255) NOT NULL"),
            ("object_type", "VARCHAR(20) NOT NULL"),
            ("display_name", "VARCHAR(255) NULL"),
            ("order_index", "INT NOT NULL DEFAULT 100"),
            ("is_visible", "TINYINT(1) NOT NULL DEFAULT 1"),
            ("last_seen_at", "DATETIME NULL"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            (
                "updated_at",
                "DATETIME DEFAULT CURRENT_TIMESTAMP "
                "ON UPDATE CURRENT_TIMESTAMP",
            ),
        ],
        "unique_indexes": {
            "uq_business_objects_project_object": [
                "project_name",
                "object_name",
            ]
        },
    },
    "business_columns": {
        "columns": [
            ("id", "INT NOT NULL AUTO_INCREMENT PRIMARY KEY"),
            ("project_name", "VARCHAR(100) NOT NULL"),
            ("object_name", "VARCHAR(255) NOT NULL"),
            ("column_name", "VARCHAR(255) NOT NULL"),
            ("data_type", "VARCHAR(64) NULL"),
            ("display_name", "VARCHAR(255) NULL"),
            ("order_index", "INT NOT NULL DEFAULT 100"),
            ("is_visible", "TINYINT(1) NOT NULL DEFAULT 1"),
            ("last_seen_at", "DATETIME NULL"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            (
                "updated_at",
                "DATETIME DEFAULT CURRENT_TIMESTAMP "
                "ON UPDATE CURRENT_TIMESTAMP",
            ),
        ],
        "unique_indexes": {
            "uq_business_columns_project_object_col": [
                "project_name",
                "object_name",
                "column_name",
            ]
        },
    },
}


def _table_exists(conn, table_name: str) -> bool:
    result = conn.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = :name "
            "LIMIT 1"
        ),
        {"name": table_name},
    ).first()
    return result is not None


def _column_names(conn, table_name: str) -> List[str]:
    rows = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = :name"
        ),
        {"name": table_name},
    ).fetchall()
    return [row[0] for row in rows]


def _unique_index_exists(conn, table_name: str, index_name: str) -> bool:
    result = conn.execute(
        text(
            "SELECT 1 FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = :table "
            "AND index_name = :idx AND non_unique = 0 "
            "LIMIT 1"
        ),
        {"table": table_name, "idx": index_name},
    ).first()
    return result is not None


def _ensure_table(
    conn,
    table_name: str,
    columns: List[Tuple[str, str]],
    unique_indexes: Dict[str, List[str]],
) -> None:
    if not _table_exists(conn, table_name):
        column_sql = ",\n    ".join(
            f"`{name}` {ddl}" for name, ddl in columns
        )
        index_sql = ""
        if unique_indexes:
            unique_parts = []
            for index_name, cols in unique_indexes.items():
                cols_sql = ", ".join(f"`{col}`" for col in cols)
                unique_parts.append(
                    f"UNIQUE KEY `{index_name}` ({cols_sql})"
                )
            index_sql = ",\n    " + ",\n    ".join(unique_parts)

        create_sql = (
            f"CREATE TABLE `{table_name}` (\n"
            f"    {column_sql}{index_sql}\n"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 "
            "COLLATE=utf8mb4_unicode_ci"
        )
        conn.execute(text(create_sql))
        print(f"[init_system_db] 已创建表 `{table_name}`。")
        return

    existing_cols = set(_column_names(conn, table_name))
    for col_name, col_ddl in columns:
        if col_name in existing_cols:
            continue
        alter_sql = (
            f"ALTER TABLE `{table_name}` "
            f"ADD COLUMN `{col_name}` {col_ddl}"
        )
        conn.execute(text(alter_sql))
        print(f"[init_system_db] 已新增列 `{table_name}`.`{col_name}`。")

    for index_name, cols in unique_indexes.items():
        if _unique_index_exists(conn, table_name, index_name):
            continue
        cols_sql = ", ".join(f"`{col}`" for col in cols)
        index_sql = (
            f"ALTER TABLE `{table_name}` "
            f"ADD UNIQUE KEY `{index_name}` ({cols_sql})"
        )
        conn.execute(text(index_sql))
        print(f"[init_system_db] 已新增唯一索引 `{index_name}`。")


def init_system_db() -> None:
    """幂等创建系统表，补齐缺失列/索引。"""
    ensure_database_exists()
    engine = get_system_engine()
    try:
        with engine.begin() as conn:
            for table_name, spec in SYSTEM_TABLES.items():
                _ensure_table(
                    conn,
                    table_name,
                    spec["columns"],  # type: ignore[arg-type]
                    spec.get("unique_indexes", {}),  # type: ignore[arg-type]
                )
    finally:
        engine.dispose()


if __name__ == "__main__":
    init_system_db()
