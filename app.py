import json
import os
import streamlit as st
import pandas as pd
from sqlalchemy import text
from pathlib import Path

# 复用你项目现有的配置
from settings import get_engine

# 从环境变量读取可选的最大表数量，默认为 5
MAX_TABLE_NUMBER = int(os.getenv("MAX_TABLE_NUMBER", "5"))

# ===========================
# 0. 核心配置 & 常量
# ===========================

SUBJECT_ID_ALIASES = [
    "SUBJECTID",   # 标准名称 (最优先)
    "SUBJID",      # 常见变体
    "patient_id",  # 外部数据常见名称
    "USUBJID"      # CDISC 标准名称 (备用)
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
    "IS NOT NULL": "不为空"
}

# ===========================
# 1. 辅助函数 (后端逻辑)
# ===========================

def load_table_metadata():
    """加载表结构信息"""
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "db" / "table_columns.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error(f"未找到表结构文件: {json_path}。请先运行 `python -m cough.db.exp_table_columns`。")
        return {}

def get_id_column(table_name, meta_data):
    """智能查找 ID 列名"""
    available_columns = meta_data.get(table_name, [])
    for alias in SUBJECT_ID_ALIASES:
        if alias in available_columns:
            return alias
    return None

@st.cache_data(ttl=600)  # 缓存10分钟，避免频繁查库
def get_unique_values(table, column, limit=100):
    """
    去数据库查询某一列的去重值（用于辅助填空）
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

def format_value_for_sql(val, operator):
    """
    根据操作符和值的类型，将其格式化为 SQL 字符串
    """
    if operator in ["IS NULL", "IS NOT NULL"]:
        return ""
    
    def is_number(s):
        try:
            float(str(s))
            return True
        except ValueError:
            return False

    # 处理 IN / NOT IN (列表)
    if operator in ["IN", "NOT IN"]:
        # 如果是 multiselect 传来的 list
        if isinstance(val, list):
            items = []
            for v in val:
                # 如果是数字，就不加引号；如果是字符串，加引号
                if is_number(v):
                    items.append(str(v))
                else:
                    items.append(f"'{v}'")
            if not items:
                return "('')" # 空列表防报错
            return f"({', '.join(items)})"
        return str(val) # 容错

    # 处理单值
    if is_number(val):
        return str(val)
    else:
        return f"'{val}'"

def build_sql(selected_tables, table_columns_map, filters, subject_blocklist, meta_data):
    """
    构建最终 SQL
    """
    if not selected_tables:
        return None

    # --- 1. 确定主表 ID ---
    base_table = selected_tables[0]
    base_id_col = get_id_column(base_table, meta_data)
    if not base_id_col:
        st.error(f"❌ 主表 `{base_table}` 中找不到 ID 列")
        return None

    # --- 2. SELECT ---
    select_clauses = []
    # 强制加上 ID 列
    select_clauses.append(f"`{base_table}`.`{base_id_col}` AS `SUBJECTID`") 

    for table in selected_tables:
        cols = table_columns_map.get(table, [])
        for col in cols:
            select_clauses.append(f"`{table}`.`{col}` AS `{table}_{col}`")

    select_sql = "SELECT\n    " + ",\n    ".join(select_clauses)

    # --- 3. FROM & JOIN ---
    from_sql = f"\nFROM `{base_table}`"
    join_sql = ""
    for i in range(1, len(selected_tables)):
        current_table = selected_tables[i]
        current_id_col = get_id_column(current_table, meta_data) or "SUBJECTID"
        join_sql += f"\nLEFT JOIN `{current_table}` ON `{base_table}`.`{base_id_col}` = `{current_table}`.`{current_id_col}`"

    # --- 4. WHERE (包含黑名单 + 可视化筛选器) ---
    where_conditions = []
    
    # 4.1 黑名单
    if subject_blocklist:
        ids = [x.strip() for x in subject_blocklist.replace("，", ",").split("\n") if x.strip()]
        if ids:
            id_list_str = "', '".join(ids)
            where_conditions.append(f"`{base_table}`.`{base_id_col}` NOT IN ('{id_list_str}')")

    # 4.2 可视化筛选器 (Condition Builder)
    if "conditions" in filters:
        for cond in filters["conditions"]:
            tbl = cond['table']
            col = cond['col']
            op = cond['op']
            val = cond['val']
            
            # 格式化值（加引号等）
            sql_val = format_value_for_sql(val, op)
            
            # 拼接: `adsl`.`AGE` > 18
            clause = f"`{tbl}`.`{col}` {op} {sql_val}"
            where_conditions.append(clause)

    where_sql = ""
    if where_conditions:
        where_sql = "\nWHERE\n  " + "\n  AND ".join(where_conditions)

    # --- 5. LIMIT (安全锁) ---
    limit_sql = "\nLIMIT 1000"

    final_sql = f"{select_sql}{from_sql}{join_sql}{where_sql}{limit_sql};"
    return final_sql

# ===========================
# 2. 界面布局 (Streamlit)
# ===========================

st.set_page_config(page_title="临床数据拼表器", layout="wide")
st.title("🏥 临床试验数据拼表工具")

meta_data = load_table_metadata()
all_tables = list(meta_data.keys())

# --- Session State 初始化 ---
# filter_rows: 存储筛选条件的列表，每项是一个 dict
if "filter_rows" not in st.session_state:
    st.session_state.filter_rows = []

def add_filter_row():
    # 添加一个空的占位符，ID 为当前长度
    st.session_state.filter_rows.append({"id": len(st.session_state.filter_rows)})

def remove_filter_row(idx):
    st.session_state.filter_rows.pop(idx)

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 全局配置")
    st.info(f"🔗 智能 Join 逻辑已启用。\nKey: {', '.join(SUBJECT_ID_ALIASES)}")
    
    st.subheader("🚫 受试者黑名单 (Not In)")
    subject_blocklist = st.text_area("输入要排除的 ID (一行一个):", height=100)

# --- 主界面 ---
st.subheader("1. 选择要拼接的表 (按 Join 顺序)")
selected_tables = st.multiselect(
    f"请选择表 (最多 {MAX_TABLE_NUMBER} 张):",
    options=all_tables,
    default=None,
    help="第一个选中的表将作为主表 (Left Table)"
)

if not selected_tables:
    st.info("👈 请先选择至少一张表。")
    st.stop()

# 限制最大选表数
if len(selected_tables) > MAX_TABLE_NUMBER:
    st.error(f"❌ 最多只能选择 {MAX_TABLE_NUMBER} 张表，当前已选 {len(selected_tables)} 张。请删除部分表。")
    st.stop()

if len(selected_tables) == MAX_TABLE_NUMBER:
    st.warning(f"⚠️ 已达到最大选表数量限制 ({MAX_TABLE_NUMBER})。")

# 显示列选择器
table_columns_map = {} 
with st.expander("2. 选择展示列 (点击展开)", expanded=True):
    cols_ui = st.columns(3)
    for idx, table_name in enumerate(selected_tables):
        # 智能提示该表的 Key
        this_id = get_id_column(table_name, meta_data)
        key_hint = f"🔑 {this_id}" if this_id else "❓ 无ID"
        
        with cols_ui[idx % 3]:
            available_cols = meta_data.get(table_name, [])
            st.markdown(f"**{table_name}** <small style='color:gray'>({key_hint})</small>", unsafe_allow_html=True)
            selected_cols = st.multiselect(
                f"选择 {table_name} 的字段",
                options=available_cols,
                default=available_cols[:5] if available_cols else [],
                key=f"sel_col_{table_name}",
                label_visibility="collapsed"
            )
            table_columns_map[table_name] = selected_cols

st.divider()

# ===========================
# 3. 可视化 WHERE 构建器
# ===========================
st.subheader("3. 筛选条件 (Where Builder)")
st.caption("构建 SQL WHERE 子句，条件之间通过 AND 连接。")

if st.button("➕ 添加筛选条件"):
    add_filter_row()

final_conditions = []

# 渲染筛选行
if st.session_state.filter_rows:
    for i, row in enumerate(st.session_state.filter_rows):
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 1])
            
            # 1. 表选择
            with c1:
                t_sel = st.selectbox("表", options=selected_tables, key=f"f_tbl_{i}", label_visibility="collapsed")
            
            # 2. 列选择 (基于表)
            with c2:
                cols = meta_data.get(t_sel, [])
                c_sel = st.selectbox("列", options=cols, key=f"f_col_{i}", label_visibility="collapsed")
            
            # 3. 操作符
            with c3:
                op_sel = st.selectbox("条件", options=list(OPERATORS.keys()), format_func=lambda x: OPERATORS[x], key=f"f_op_{i}", label_visibility="collapsed")
            
            # 4. 值输入 (根据操作符变化)
            with c4:
                val_key = f"f_val_{i}"
                
                # 特殊逻辑：如果是 IN / NOT IN，显示多选框，并尝试加载数据
                if op_sel in ["IN", "NOT IN"]:
                    # 使用 session_state 保存每一行已加载的候选值，保证多次交互后仍能回显
                    loaded_vals_key = f"loaded_vals_{i}"
                    loaded_vals = st.session_state.get(loaded_vals_key, [])

                    # 加载值的功能放在一个小的 expander 里以免占据太多空间
                    with st.expander("🔍 加载值", expanded=False):
                        if st.button("从数据库加载 Top 100", key=f"btn_load_{i}"):
                            loaded_vals = get_unique_values(t_sel, c_sel)
                            # 将加载结果持久化到 session_state，避免下次交互丢失
                            st.session_state[loaded_vals_key] = loaded_vals
                            # 简要提示加载结果
                            if loaded_vals:
                                st.success(f"已加载 {len(loaded_vals)} 个值，请在下方选择或输入。")

                    # 为了保证“已选择的值”在 options 中始终可见，
                    # 将当前选中的值与已加载的候选值合并去重后作为 options
                    current_selected = st.session_state.get(val_key, [])
                    # 转成字符串，保持与 loaded_vals 类型一致
                    current_selected = [str(v) for v in current_selected]
                    merged_options = sorted(set(current_selected) | set(loaded_vals))

                    val_input = st.multiselect(
                        "值", 
                        options=merged_options, 
                        key=val_key,
                        label_visibility="collapsed",
                        placeholder="输入值并回车，或选择..."
                    )
                
                elif op_sel in ["IS NULL", "IS NOT NULL"]:
                    val_input = None
                    st.write("---")
                
                else:
                    # 单值输入
                    val_input = st.text_input("值", key=val_key, label_visibility="collapsed")

            # 5. 删除
            with c5:
                if st.button("🗑️", key=f"btn_del_{i}"):
                    remove_filter_row(i)
                    st.rerun()

            # 收集有效条件
            if t_sel and c_sel and op_sel:
                if (op_sel in ["IS NULL", "IS NOT NULL"]) or val_input:
                    final_conditions.append({
                        "table": t_sel,
                        "col": c_sel,
                        "op": op_sel,
                        "val": val_input
                    })
else:
    st.info("暂无筛选条件。点击上方按钮添加。")

filters_config = {"conditions": final_conditions}

# --- 生成 ---
st.divider()

if st.button("🚀 生成 SQL 并预览数据", type="primary"):
    sql = build_sql(selected_tables, table_columns_map, filters_config, subject_blocklist, meta_data)
    
    if sql:
        st.subheader("生成的 SQL:")
        st.code(sql, language="sql")
        
        try:
            with st.spinner("正在查询..."):
                engine = get_engine()
                # 加上 execution_options(timeout=30) 防止卡死
                with engine.connect().execution_options(timeout=60) as conn:
                    df_result = pd.read_sql(sql, conn)
                
            st.success(f"查询成功！预览前 {len(df_result)} 行 (已限制 Limit 1000)。")
            st.dataframe(df_result, use_container_width=True)
            
            # 只有当有数据时才显示下载
            if not df_result.empty:
                csv = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载结果为 CSV",
                    data=csv,
                    file_name="cohort_data.csv",
                    mime="text/csv",
                )
            
        except Exception as e:
            st.error(f"SQL 执行错误: {e}")
            st.warning("提示: 如果查询超时，请尝试减少选择的表数量或增加筛选条件。")
    else:
        st.error("无法生成 SQL，请检查配置。")
