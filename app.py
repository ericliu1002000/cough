import json
import streamlit as st
import pandas as pd
from sqlalchemy import text
from pathlib import Path

# 复用你项目现有的配置
from settings import get_engine

# ===========================
# 0. 核心配置 (Config)
# ===========================

# 定义受试者 ID 的“别名列表” (按优先级查找)
# 系统会依次检查表中是否存在这些列，找到第一个存在的就用它作为 Join Key
SUBJECT_ID_ALIASES = [
    "SUBJECTID",   # 标准名称 (最优先)
    "SUBJID",      # 常见变体
    "patient_id",  # 外部数据常见名称
    "USUBJID"      # CDISC 标准名称 (备用)
]

# ===========================
# 1. 辅助函数
# ===========================

def load_table_metadata():
    """加载表结构信息"""
    # 使用相对于当前 app.py 的路径，避免找不到文件
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "db" / "table_columns.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error(f"未找到表结构文件: {json_path}。请先运行 `python -m cough.db.exp_table_columns`。")
        return {}

def get_id_column(table_name, meta_data):
    """
    智能查找：根据配置的别名列表，找到该表实际使用的 ID 列名。
    如果没找到，返回 None。
    """
    available_columns = meta_data.get(table_name, [])
    
    for alias in SUBJECT_ID_ALIASES:
        if alias in available_columns:
            return alias
            
    return None

def build_sql(selected_tables, table_columns_map, filters, subject_blocklist, meta_data):
    """
    核心：根据用户的选择，动态拼接 SQL 语句 (支持智能 ID 映射)
    """
    if not selected_tables:
        return None

    # --- 1. 确定主表的 ID 列 ---
    base_table = selected_tables[0]
    base_id_col = get_id_column(base_table, meta_data)
    
    if not base_id_col:
        st.error(f"❌ 主表 `{base_table}` 中找不到任何已知的 ID 列 ({SUBJECT_ID_ALIASES})，无法作为主表。")
        return None

    # --- 2. 构建 SELECT 部分 ---
    select_clauses = []
    
    # 强制把主表的 ID 选出来，并统一重命名为 'SUBJECTID' 方便查看
    select_clauses.append(f"`{base_table}`.`{base_id_col}` AS `SUBJECTID`") 

    for table in selected_tables:
        cols = table_columns_map.get(table, [])
        for col in cols:
            # 如果这一列就是该表的 ID 列，我们跳过（因为已经强制加在第一列了），或者你可以保留但改名
            # 这里简单起见：保留，命名为 Table_Col
            select_clauses.append(f"`{table}`.`{col}` AS `{table}_{col}`")

    select_sql = "SELECT\n    " + ",\n    ".join(select_clauses)

    # --- 3. 构建 FROM 和 LEFT JOIN 部分 ---
    from_sql = f"\nFROM `{base_table}`"
    
    join_sql = ""
    # 从第二个表开始遍历
    for i in range(1, len(selected_tables)):
        current_table = selected_tables[i]
        
        # 找到当前这个表的 ID 列名
        current_id_col = get_id_column(current_table, meta_data)
        
        if not current_id_col:
            st.warning(f"⚠️ 表 `{current_table}` 中找不到 ID 列，将无法正确 Join (SQL 中会留空，请手动检查)。")
            # 降级处理：还是默认 SUBJECTID，防止 SQL 彻底报错无法生成
            current_id_col = "SUBJECTID" 

        # 逻辑：LEFT JOIN TableB ON BaseTable.BaseID = TableB.CurrentID
        join_sql += f"\nLEFT JOIN `{current_table}` ON `{base_table}`.`{base_id_col}` = `{current_table}`.`{current_id_col}`"

    # --- 4. 构建 WHERE 部分 ---
    where_conditions = []
    
    # 4.1 处理黑名单 (使用主表的 ID 列)
    if subject_blocklist:
        ids = [x.strip() for x in subject_blocklist.replace("，", ",").split("\n") if x.strip()]
        if ids:
            id_list_str = "', '".join(ids)
            where_conditions.append(f"`{base_table}`.`{base_id_col}` NOT IN ('{id_list_str}')")

    # 4.2 自定义 WHERE
    if filters.get("custom_where"):
        where_conditions.append(filters["custom_where"])

    where_sql = ""
    if where_conditions:
        where_sql = "\nWHERE " + "\n  AND ".join(where_conditions)

    # --- 5. 构建 GROUP BY / HAVING ---
    group_by_sql = ""
    if filters.get("group_by"):
        # 还原列名 (Table_Column -> Table.Column)
        # 这里的处理比较简单，假设用户选的都是标准生成的 Table_Col
        group_cols_sql = []
        for c in filters["group_by"]:
            # c 的格式是 "TableName_ColName"
            # 我们需要反向找到它属于哪个表。最简单的方法是拆分字符串，但这有风险（如果表名带下划线）。
            # 更稳妥的方法是去 table_columns_map 里查。
            found = False
            for tbl, t_cols in table_columns_map.items():
                for t_col in t_cols:
                    if f"{tbl}_{t_col}" == c:
                        group_cols_sql.append(f"`{tbl}`.`{t_col}`")
                        found = True
                        break
                if found: break
        
        if group_cols_sql:
            group_by_sql = "\nGROUP BY " + ", ".join(group_cols_sql)

    having_sql = ""
    if filters.get("having"):
        having_sql = "\nHAVING " + filters["having"]

    # --- 6. 组装 ---
    final_sql = f"{select_sql}{from_sql}{join_sql}{where_sql}{group_by_sql}{having_sql};"
    return final_sql

# ===========================
# 2. 界面布局 (Streamlit)
# ===========================

st.set_page_config(page_title="临床数据拼表器", layout="wide")
st.title("🏥 临床试验数据拼表工具")

# 加载元数据
meta_data = load_table_metadata()
all_tables = list(meta_data.keys())

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 全局配置")
    
    # 显示当前的 ID 映射规则
    st.info(f"🔗 智能 Join 逻辑已启用。\n\n系统将按以下优先顺序查找各表的关联键：\n\n" + " -> ".join(SUBJECT_ID_ALIASES))
    
    st.subheader("🚫 受试者黑名单 (Not In)")
    subject_blocklist = st.text_area(
        "输入要排除的 ID (一行一个):",
        height=150,
        placeholder="1001\n1002"
    )

# --- 主界面 ---

st.subheader("1. 选择要拼接的表 (按 Join 顺序)")
selected_tables = st.multiselect(
    "请选择表 (第一个选中的将作为主表):",
    options=all_tables,
    default=None
)

if not selected_tables:
    st.warning("请至少选择一张表。")
    st.stop()

# 实时检查主表的 ID
main_tbl = selected_tables[0]
main_id = get_id_column(main_tbl, meta_data)
if main_id:
    st.success(f"✅ 主表 `{main_tbl}` 将使用 `{main_id}` 作为关联主键。")
else:
    st.error(f"❌ 警告：在主表 `{main_tbl}` 中未找到配置的 ID 列，请检查表结构或修改配置。")

st.subheader("2. 选择每张表要展示的列")
table_columns_map = {} 
cols = st.columns(len(selected_tables))
all_selected_columns_ref = [] 

for idx, table_name in enumerate(selected_tables):
    available_cols = meta_data.get(table_name, [])
    
    # 标注一下该表用的是哪个 ID
    this_id = get_id_column(table_name, meta_data)
    id_label = f" (Key: {this_id})" if this_id else " (Key: ❓)"
    
    with st.expander(f"表: {table_name}{id_label}", expanded=True):
        default_cols = available_cols[:5] if len(available_cols) > 0 else []
        selected_cols = st.multiselect(
            f"选择字段:",
            options=available_cols,
            default=default_cols,
            key=f"select_{table_name}"
        )
        table_columns_map[table_name] = selected_cols
        for c in selected_cols:
            all_selected_columns_ref.append(f"{table_name}_{c}")

# --- 高级筛选 ---
st.subheader("3. 高级筛选 (SQL)")
col1, col2, col3 = st.columns(3)
filters = {}
with col1:
    filters["custom_where"] = st.text_input("WHERE 条件", placeholder="例如: `adsl`.`AGE` > 18")
with col2:
    filters["group_by"] = st.multiselect("GROUP BY 字段", options=all_selected_columns_ref)
with col3:
    filters["having"] = st.text_input("HAVING 条件", placeholder="例如: count(*) > 1")

# --- 生成 ---
st.divider()

if st.button("🚀 生成大表并预览", type="primary"):
    # 传入 meta_data 以供查询列名
    sql = build_sql(selected_tables, table_columns_map, filters, subject_blocklist, meta_data)
    
    if sql:
        st.subheader("生成的 SQL 语句:")
        st.code(sql, language="sql")
        
        try:
            engine = get_engine()
            with st.spinner("正在从数据库查询数据..."):
                df_result = pd.read_sql(sql, engine)
                
            st.success(f"查询成功！共找到 {len(df_result)} 行数据。")
            st.dataframe(df_result, use_container_width=True)
            
            csv = df_result.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下载结果为 CSV",
                data=csv,
                file_name="cohort_data.csv",
                mime="text/csv",
            )
            
        except Exception as e:
            st.error(f"查询出错: {e}")
    else:
        st.error("生成失败，请检查配置。")