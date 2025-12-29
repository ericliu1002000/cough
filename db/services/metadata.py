"""业务库元数据同步与维护服务（系统库）。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from sqlalchemy import inspect, text

from analysis.settings.config import get_business_engine, get_system_engine
from db.services.db_config import get_business_db_config

DEFAULT_ORDER_INDEX = 100


def _normalize_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _normalize_order_index(value: Optional[object]) -> int:
    normalized = _normalize_int(value)
    return normalized if normalized is not None else DEFAULT_ORDER_INDEX


def _normalize_str(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text_val = str(value).strip()
    return text_val or None


def sync_business_metadata(project_name: Optional[str] = None) -> Dict[str, int]:
    """
    扫描业务库的表/视图与列信息，并写入系统库元数据表。
    仅新增或更新类型信息，不覆盖人工配置。
    """
    config = get_business_db_config()
    project = project_name or config["code"]
    db_name = config["database"]

    business_engine = get_business_engine()
    system_engine = get_system_engine()

    inspector = inspect(business_engine)
    table_names = inspector.get_table_names(schema=db_name)
    view_names = inspector.get_view_names(schema=db_name)

    object_rows: List[Dict[str, object]] = []
    column_rows: List[Dict[str, object]] = []

    for name in table_names:
        object_rows.append(
            {
                "project_name": project,
                "object_name": name,
                "object_type": "table",
                "order_index": DEFAULT_ORDER_INDEX,
            }
        )

    for name in view_names:
        object_rows.append(
            {
                "project_name": project,
                "object_name": name,
                "object_type": "view",
                "order_index": DEFAULT_ORDER_INDEX,
            }
        )

    for object_name in table_names + view_names:
        try:
            columns = inspector.get_columns(object_name, schema=db_name)
        except Exception as exc:
            raise RuntimeError(
                f"读取列信息失败: {object_name}，原因: {exc}"
            ) from exc
        for column in columns:
            col_name = column.get("name")
            if not col_name:
                continue
            data_type = column.get("type")
            column_rows.append(
                {
                    "project_name": project,
                    "object_name": object_name,
                    "column_name": col_name,
                    "data_type": str(data_type) if data_type is not None else None,
                    "order_index": DEFAULT_ORDER_INDEX,
                }
            )

    objects_inserted = 0
    columns_inserted = 0

    if object_rows:
        insert_sql = text(
            "INSERT INTO business_objects "
            "(project_name, object_name, object_type, order_index, is_visible, last_seen_at) "
            "VALUES (:project_name, :object_name, :object_type, :order_index, 1, NOW()) "
            "ON DUPLICATE KEY UPDATE "
            "object_type = VALUES(object_type), "
            "updated_at = NOW(), "
            "last_seen_at = NOW()"
        )
        with system_engine.begin() as conn:
            result = conn.execute(insert_sql, object_rows)
            objects_inserted = result.rowcount or 0

    if column_rows:
        insert_sql = text(
            "INSERT INTO business_columns "
            "(project_name, object_name, column_name, data_type, order_index, is_visible, last_seen_at) "
            "VALUES (:project_name, :object_name, :column_name, :data_type, :order_index, 1, NOW()) "
            "ON DUPLICATE KEY UPDATE "
            "data_type = VALUES(data_type), "
            "updated_at = NOW(), "
            "last_seen_at = NOW()"
        )
        with system_engine.begin() as conn:
            result = conn.execute(insert_sql, column_rows)
            columns_inserted = result.rowcount or 0

    business_engine.dispose()
    system_engine.dispose()

    return {
        "objects_scanned": len(object_rows),
        "columns_scanned": len(column_rows),
        "objects_upserted": objects_inserted,
        "columns_upserted": columns_inserted,
    }


def fetch_business_objects(
    project_name: str, include_hidden: bool = True
) -> List[Dict[str, object]]:
    """读取业务对象（表/视图）列表。"""
    sql = (
        "SELECT id, object_name, object_type, display_name, order_index, "
        "is_visible, last_seen_at "
        "FROM business_objects "
        "WHERE project_name = :project "
    )
    if not include_hidden:
        sql += "AND is_visible = 1 "
    sql += (
        "ORDER BY "
        f"COALESCE(order_index, {DEFAULT_ORDER_INDEX}) DESC, object_name"
    )

    engine = get_system_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"project": project_name}).mappings().all()
    engine.dispose()
    return [dict(row) for row in rows]


def fetch_business_columns(
    project_name: str,
    object_name: str,
    include_hidden: bool = True,
) -> List[Dict[str, object]]:
    """读取指定对象的列配置列表。"""
    sql = (
        "SELECT id, column_name, data_type, display_name, order_index, "
        "is_visible, last_seen_at "
        "FROM business_columns "
        "WHERE project_name = :project AND object_name = :object "
    )
    if not include_hidden:
        sql += "AND is_visible = 1 "
    sql += (
        "ORDER BY "
        f"COALESCE(order_index, {DEFAULT_ORDER_INDEX}) DESC, column_name"
    )

    engine = get_system_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(sql), {"project": project_name, "object": object_name}
        ).mappings().all()
    engine.dispose()
    return [dict(row) for row in rows]


def update_business_objects(
    project_name: str,
    rows: Iterable[Dict[str, object]],
) -> int:
    """批量更新表/视图配置。"""
    sql = text(
        "UPDATE business_objects "
        "SET display_name = :display_name, "
        "    order_index = :order_index, "
        "    is_visible = :is_visible "
        "WHERE id = :id AND project_name = :project"
    )
    payload = []
    for row in rows:
        payload.append(
            {
                "id": row.get("id"),
                "project": project_name,
                "display_name": _normalize_str(row.get("display_name")),
                "order_index": _normalize_order_index(row.get("order_index")),
                "is_visible": 1 if row.get("is_visible") else 0,
            }
        )

    engine = get_system_engine()
    with engine.begin() as conn:
        result = conn.execute(sql, payload)
    engine.dispose()
    return result.rowcount or 0


def update_business_columns(
    project_name: str,
    object_name: str,
    rows: Iterable[Dict[str, object]],
) -> int:
    """批量更新列配置。"""
    sql = text(
        "UPDATE business_columns "
        "SET display_name = :display_name, "
        "    order_index = :order_index, "
        "    is_visible = :is_visible "
        "WHERE id = :id AND project_name = :project AND object_name = :object"
    )
    payload = []
    for row in rows:
        payload.append(
            {
                "id": row.get("id"),
                "project": project_name,
                "object": object_name,
                "display_name": _normalize_str(row.get("display_name")),
                "order_index": _normalize_order_index(row.get("order_index")),
                "is_visible": 1 if row.get("is_visible") else 0,
            }
        )

    engine = get_system_engine()
    with engine.begin() as conn:
        result = conn.execute(sql, payload)
    engine.dispose()
    return result.rowcount or 0


def update_business_column_display_names(
    project_name: str,
    object_name: str,
    display_map: Dict[str, str],
    override: bool = False,
) -> int:
    """按列名批量更新显示名（可选覆盖已有配置）。"""
    if not display_map:
        return 0

    condition = ""
    if not override:
        condition = "AND (display_name IS NULL OR display_name = '') "

    sql = text(
        "UPDATE business_columns "
        "SET display_name = :display_name "
        "WHERE project_name = :project "
        "AND object_name = :object "
        "AND column_name = :column "
        f"{condition}"
    )
    payload = []
    for column_name, display_name in display_map.items():
        normalized = _normalize_str(display_name)
        if not normalized:
            continue
        payload.append(
            {
                "project": project_name,
                "object": object_name,
                "column": column_name,
                "display_name": normalized,
            }
        )

    if not payload:
        return 0

    engine = get_system_engine()
    with engine.begin() as conn:
        result = conn.execute(sql, payload)
    engine.dispose()
    return result.rowcount or 0


def get_table_columns_map(
    project_name: Optional[str] = None,
    include_hidden: bool = False,
) -> Dict[str, List[str]]:
    """读取表->列映射（支持可见性过滤）。"""
    config = get_business_db_config()
    project = project_name or config["code"]

    objects_sql = (
        "SELECT object_name "
        "FROM business_objects "
        "WHERE project_name = :project "
    )
    columns_sql = (
        "SELECT object_name, column_name "
        "FROM business_columns "
        "WHERE project_name = :project "
    )
    if not include_hidden:
        objects_sql += "AND is_visible = 1 "
        columns_sql += "AND is_visible = 1 "
    objects_sql += (
        f"ORDER BY COALESCE(order_index, {DEFAULT_ORDER_INDEX}) DESC, object_name"
    )
    columns_sql += (
        f"ORDER BY COALESCE(order_index, {DEFAULT_ORDER_INDEX}) DESC, column_name"
    )

    engine = get_system_engine()
    with engine.connect() as conn:
        objects = conn.execute(
            text(objects_sql), {"project": project}
        ).mappings().all()
        columns = conn.execute(
            text(columns_sql), {"project": project}
        ).mappings().all()
    engine.dispose()

    table_columns: Dict[str, List[str]] = {
        row["object_name"]: [] for row in objects
    }
    for row in columns:
        object_name = row["object_name"]
        if object_name not in table_columns:
            continue
        table_columns[object_name].append(row["column_name"])

    return table_columns
