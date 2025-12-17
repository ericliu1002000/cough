import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from sqlalchemy import text

from settings import get_engine

# ===========================
# 核心配置 & 常量
# ===========================

SUBJECT_ID_ALIASES = [
    "SUBJID",  # 常见变体
    "USUBJID",  # CDISC 标准名称 (备用)
    "SUBJECTID",  # 标准名称 (最优先)
    "patient_id",  # 外部数据常见名称
]

OPERATORS = {
    "=": "等于 (=)",
    ">": "大于 (>)",
    "<": "小于 (<)",
    ">=": "大于等于 (>=)",
    "<=": "小于等于 (<=)",
    "!=": "不等于 (!=)",
    "IN": "包含于 (IN)",
    "NOT IN": "不包含 (NOT IN)",
    "LIKE": "像 (LIKE)",
    "IS NULL": "为空",
    "IS NOT NULL": "不为空",
}


# ===========================
# 元数据与 ID 解析
# ===========================

def load_table_metadata() -> Dict[str, List[str]]:
    """
    加载表结构信息 (db/table_columns.json)。

    返回:
        {table_name: [col1, col2, ...], ...}
    """
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "db" / "table_columns.json"

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error(
            f"未找到表结构文件: {json_path}。请先运行 `python -m cough.db.exp_table_columns`。"
        )
        return {}


def get_id_column(table_name: str, meta_data: Dict[str, List[str]]) -> str | None:
    """
    智能查找 ID 列名:
    - 按 SUBJECT_ID_ALIASES 的优先级，在该表的列中查找第一个匹配的列名。
    - 若未找到，返回 None。
    """
    available_columns = meta_data.get(table_name, [])
    for alias in SUBJECT_ID_ALIASES:
        if alias in available_columns:
            return alias
    return None


# ===========================
# 分析集配置 CRUD
# ===========================

def fetch_all_setups() -> List[Dict[str, Any]]:
    """
    从 analysis_list_setups 表读取所有已保存的配置列表。

    返回:
        [{'id': ..., 'setup_name': ..., 'description': ...}, ...]
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
    except Exception as e:
        print(f"[Warning] 无法加载分析集配置列表: {e}")
        return []


def fetch_setup_config(setup_name: str) -> Dict[str, Any] | None:
    """
    根据 setup_name 读取单个配置的一段/二段配置。

    返回:
        {
          "extraction": {...},        # 一段配置 (dict)
          "calculation": ... or None # 二段配置 (任意 JSON，可为 None)
        }
        如果不存在记录，则返回 None。
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

        def _normalize_json(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        extraction = _normalize_json(row["config_extraction"])
        raw_calculation = _normalize_json(row["config_calculation"])

        # 统一二段配置的 JSON 结构，保证至少包含：
        # - calc_rules: List[Dict]
        # - note: str
        # - exclusions: List[Dict]
        # - pivot: Dict[str, Any]
        # 同时兼容历史数据（仅存 List 或 None）。
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
        elif isinstance(raw_calculation, list):
            # 早期版本只保存了规则列表
            calculation = {
                "calc_rules": raw_calculation,
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
            # 复制一份，补齐默认键，保留未来可能增加的字段（如 pivot、exclusions）
            calculation = dict(raw_calculation)
            calculation.setdefault("calc_rules", [])
            calculation.setdefault("note", "")
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
        return None


def save_setup_config(setup_name: str, description: str | None, config: Dict[str, Any]) -> None:
    """兼容旧接口，内部转调 save_extraction_config。"""
    save_extraction_config(setup_name, description, config)


def save_extraction_config(
    setup_name: str, description: str | None, config_data: Dict[str, Any]
) -> None:
    """
    保存或更新「一段配置」到 analysis_list_setups.config_extraction。

    逻辑：
    - 若记录不存在：插入新记录，config_calculation 置为 NULL；
    - 若记录存在：仅更新 description 与 config_extraction，不修改 config_calculation。
    """
    config_json = json.dumps(config_data, ensure_ascii=False)
    engine = get_engine()
    sql = text(
        """
        INSERT INTO analysis_list_setups (setup_name, description, config_extraction, config_calculation, created_at, updated_at)
        VALUES (:name, :desc, :config_extraction, NULL, NOW(), NOW())
        ON DUPLICATE KEY UPDATE
            description = VALUES(description),
            config_extraction = VALUES(config_extraction),
            updated_at = NOW()
        """
    )
    try:
        with engine.connect() as conn:
            conn.execute(
                sql,
                {
                    "name": setup_name,
                    "desc": description,
                    "config_extraction": config_json,
                },
            )
            conn.commit()
    except Exception as e:
        st.error(f"保存一段配置 `{setup_name}` 失败: {e}")
    finally:
        engine.dispose()


def save_calculation_config(setup_name: str, calculation_data: Any) -> None:
    """
    保存或更新「二段配置」到 analysis_list_setups.config_calculation。

    仅更新 config_calculation 字段，不修改一段配置。
    calculation_data 建议为 dict，例如：
        {
          "calc_rules": [...],
          "note": "本次分析备注",
          ...  # 未来可扩展 pivot / exclusions 等
        }
    """
    config_json = json.dumps(calculation_data, ensure_ascii=False)
    engine = get_engine()
    sql = text(
        """
        UPDATE analysis_list_setups
        SET config_calculation = :config_calculation,
            updated_at = NOW()
        WHERE setup_name = :name
        """
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
    finally:
        engine.dispose()


def delete_setup_config(setup_name: str) -> None:
    """
    删除指定名称的分析集配置。
    """
    engine = get_engine()
    sql = text("DELETE FROM analysis_list_setups WHERE setup_name = :name")
    try:
        with engine.connect() as conn:
            conn.execute(sql, {"name": setup_name})
            conn.commit()
    except Exception as e:
        st.error(f"删除配置 `{setup_name}` 失败: {e}")
    finally:
        engine.dispose()


# ===========================
# 数据值与 SQL 构建
# ===========================

@st.cache_data(ttl=600)  # 缓存10分钟，避免频繁查库
def get_unique_values(table: str, column: str, limit: int = 100) -> List[str]:
    """
    去数据库查询某一列的去重值（用于辅助填空）。
    """
    try:
        engine = get_engine()
        # 加上反引号防止关键字冲突
        query = f"SELECT DISTINCT `{column}` FROM `{table}` LIMIT {limit}"
        df = pd.read_sql(query, engine)
        # 将结果转为列表，过滤空值
        values = df.iloc[:, 0].dropna().astype(str).tolist()
        return sorted(values)
    except Exception as e:
        # 不阻塞主流程，只在后台记录
        print(f"[Warning] 无法获取列值: {e}")
        return []


def format_value_for_sql(val: Any, operator: str) -> str:
    """
    根据操作符和值的类型，将其格式化为 SQL 字符串。
    """
    if operator in ["IS NULL", "IS NOT NULL"]:
        return ""

    def is_number(s: Any) -> bool:
        try:
            float(str(s))
            return True
        except ValueError:
            return False

    # 处理 IN / NOT IN (列表)
    if operator in ["IN", "NOT IN"]:
        # 如果是 multiselect 传来的 list
        if isinstance(val, list):
            items: List[str] = []
            for v in val:
                # 如果是数字，就不加引号；如果是字符串，加引号
                if is_number(v):
                    items.append(str(v))
                else:
                    items.append(f"'{v}'")
            if not items:
                return "('')"  # 空列表防报错
            return f"({', '.join(items)})"
        return str(val)  # 容错

    # 处理单值
    if is_number(val):
        return str(val)
    else:
        return f"'{val}'"


def build_sql(
    selected_tables: List[str],
    table_columns_map: Dict[str, List[str]],
    filters: Dict[str, Any],
    subject_blocklist: str,
    meta_data: Dict[str, List[str]],
    group_by: List[Dict[str, Any]] | None = None,
    aggregations: List[Dict[str, Any]] | None = None,
) -> str | None:
    """
    构建最终 SQL。

    支持两种模式：
    1) 行级模式（默认，无 group_by）：SELECT 主键 + 所有选中列，不做聚合；
    2) 聚合模式（配置了 group_by）：SELECT 仅包含 group_by 字段和 aggregations 中的聚合表达式，并添加 GROUP BY。
    """
    if not selected_tables:
        return None

    # 规范化分组与聚合配置
    group_by = group_by or []
    aggregations = aggregations or []
    use_group_mode = len(group_by) > 0

    # --- 1. 确定主表 ID ---
    base_table = selected_tables[0]
    base_id_col = get_id_column(base_table, meta_data)
    if not base_id_col:
        st.error(f"❌ 主表 `{base_table}` 中找不到 ID 列")
        return None

    # --- 2. SELECT ---
    select_clauses: List[str] = []

    if not use_group_mode:
        # 行级模式：强制加上 ID 列 + 所有选中列
        select_clauses.append(f"`{base_table}`.`{base_id_col}` AS `SUBJECTID`")

        for table in selected_tables:
            cols = table_columns_map.get(table, [])
            for col in cols:
                select_clauses.append(f"`{table}`.`{col}` AS `{table}_{col}`")
    else:
        # 聚合模式
        # 2.1 分组字段进入 SELECT 与 GROUP BY
        for item in group_by:
            tbl = item.get("table")
            col = item.get("col")
            if not tbl or not col:
                continue
            alias = item.get("alias") or f"{tbl}_{col}"
            select_clauses.append(f"`{tbl}`.`{col}` AS `{alias}`")

        # 2.2 聚合字段
        for agg in aggregations:
            tbl = agg.get("table")
            col = agg.get("col")
            func = (agg.get("func") or "").upper()
            if not tbl or not col or not func:
                continue
            alias = agg.get("alias") or f"{func}_{tbl}_{col}"
            # 简单防注入：仅允许字母、数字、下划线的函数名
            if not func.replace("_", "").isalnum():
                continue
            select_clauses.append(f"{func}(`{tbl}`.`{col}`) AS `{alias}`")

        # 如果在聚合模式下，没有任何字段最终进入 SELECT，认为配置有误
        if not select_clauses:
            st.error("已启用 Group By，但未配置任何分组字段或聚合字段，无法生成 SQL。")
            return None

    select_sql = "SELECT\n    " + ",\n    ".join(select_clauses)

    # --- 3. FROM & JOIN ---
    from_sql = f"\nFROM `{base_table}`"
    join_sql = ""
    for i in range(1, len(selected_tables)):
        current_table = selected_tables[i]
        current_id_col = get_id_column(current_table, meta_data) or "SUBJECTID"
        join_sql += (
            f"\nLEFT JOIN `{current_table}` "
            f"ON `{base_table}`.`{base_id_col}` = `{current_table}`.`{current_id_col}`"
        )

    # --- 4. WHERE (包含黑名单 + 可视化筛选器) ---
    where_conditions: List[str] = []

    # 4.1 黑名单
    if subject_blocklist:
        ids = [
            x.strip()
            for x in subject_blocklist.replace("，", ",").split("\n")
            if x.strip()
        ]
        if ids:
            id_list_str = "', '".join(ids)
            where_conditions.append(
                f"`{base_table}`.`{base_id_col}` NOT IN ('{id_list_str}')"
            )

    # 4.2 可视化筛选器 (Condition Builder)
    if "conditions" in filters:
        for cond in filters["conditions"]:
            tbl = cond["table"]
            col = cond["col"]
            op = cond["op"]
            val = cond["val"]

            # 格式化值（加引号等）
            sql_val = format_value_for_sql(val, op)

            # 拼接: `adsl`.`AGE` > 18
            clause = f"`{tbl}`.`{col}` {op} {sql_val}"
            where_conditions.append(clause)

    where_sql = ""
    if where_conditions:
        where_sql = "\nWHERE\n  " + "\n  AND ".join(where_conditions)

    # --- 5. GROUP BY（仅聚合模式） ---
    group_by_sql = ""
    if use_group_mode and group_by:
        gb_parts: List[str] = []
        for item in group_by:
            tbl = item.get("table")
            col = item.get("col")
            if not tbl or not col:
                continue
            gb_parts.append(f"`{tbl}`.`{col}`")
        if gb_parts:
            group_by_sql = "\nGROUP BY\n  " + ",\n  ".join(gb_parts)

    # --- 6. LIMIT (安全锁) ---
    limit_sql = "\nLIMIT 1000"

    final_sql = f"{select_sql}{from_sql}{join_sql}{where_sql}{group_by_sql}{limit_sql};"
    return final_sql
