import pandas as pd
import numpy as np
import re
import os

# ===========================
# 配置常量
# ===========================
TYPE1_COLS = [
    "生育期", "比对照±天", "株高", "穗位", "倒伏率", "倒折率", 
    "穗长", "穗粗", "秃尖长", "穗行数", "行粒数", "百粒重", 
    "折亩产量", "邻ck亩产量", "比邻ck±%"
]

TYPE2_COLS = [
    "大斑病", "丝黑穗病", "茎腐病", "穗腐病", "矮花叶病"
]

TYPE3_COLS = [
    "穗型", "轴色", "粒型", "粒色"
]

def parse_value(val):
    """
    清洗数值逻辑：
    1. 单元格未填或写“无”写的视为：0
    2. 单元格填I的， 视为1
    3. 单元格写类似 16-18的， 视为 16和18的平均值
    4. 写17.2（16-18）的， 直接忽略（）内的内容，视为17.2
    """
    if pd.isna(val) or val == "":
        return 0.0
    
    s_val = str(val).strip()
    
    if not s_val or s_val == "无":
        return 0.0
    
    if s_val == "I":
        return 1.0
    
    # 处理括号：17.2（16-18） -> 17.2 (兼容中文和英文括号)
    s_val = re.sub(r'[\(（].*?[\)）]', '', s_val).strip()
    
    # 处理范围：16-18 -> 平均值
    # 简单判断是否包含 '-' 且不是负号开头
    if '-' in s_val and not s_val.startswith('-'):
        parts = s_val.split('-')
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if nums:
            return sum(nums) / len(nums)
            
    try:
        return float(s_val)
    except ValueError:
        return 0.0

def get_mode(series):
    """获取众数"""
    valid = series.dropna()
    # 过滤空字符串
    valid = valid[valid.astype(str).str.strip() != ""]
    if valid.empty:
        return None
    # mode() 可能返回多个，取第一个
    m = valid.mode()
    return m.iloc[0] if not m.empty else None

def get_stats_dict(file_path: str) -> dict:
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return {}

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return {}

    if df.empty:
        print("数据为空")
        return {}

    # 1. 按照第一列进行分类
    group_col = df.columns[0]
    agg_rules = {}

    # 应用规则
    for col in TYPE1_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_value)
            agg_rules[col] = 'mean'
    
    for col in TYPE2_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_value)
            agg_rules[col] = 'max'
            
    for col in TYPE3_COLS:
        if col in df.columns:
            agg_rules[col] = get_mode

    if not agg_rules:
        print("未找到指定列")
        return {}

    result = df.groupby(group_col, as_index=False).agg(agg_rules)

    # 统计增产点率
    yield_stats = {}
    if "比邻ck±%" in df.columns:
        for name, group in df.groupby(group_col):
            vals = group["比邻ck±%"]
            total = len(vals)
            inc = (vals > 0).sum()
            if total > 0:
                rate = (inc / total) * 100
                rate_str = f"{rate:.0f}" if rate.is_integer() else f"{rate:.1f}"
                yield_stats[name] = f"{total}点试验{inc}点增产，增产点率{rate_str}%"

    output_order = [
        "生育期", "比对照±天", "株高", "穗位", "倒伏率", "倒折率", 
        "大斑病", "丝黑穗病", "茎腐病", "穗腐病", "矮花叶病", 
        "穗长", "穗粗", "秃尖长", "穗行数", "行粒数", "百粒重", 
        "穗型", "轴色", "粒型", "粒色", 
        "折亩产量", "邻ck亩产量", "比邻ck±%"
    ]

    # 配置映射：(模板名称, 单位)
    col_config = {
        "生育期": ("生育期", "天"),
        "比对照±天": ("比对照", "天"), 
        "株高": ("株高", "cm"),
        "穗位": ("穗位", "cm"),
        "倒伏率": ("倒伏率", "%"),
        "倒折率": ("倒折率", "%"),
        "大斑病": ("大斑病", "级"),
        "丝黑穗病": ("丝黑穗病", "%"),
        "茎腐病": ("茎腐病", "%"),
        "穗腐病": ("穗腐病", "%"),
        "矮花叶病": ("矮花叶病", "级"),
        "穗长": ("穗长", "cm"),
        "穗粗": ("穗粗", "cm"),
        "秃尖长": ("秃尖长", "cm"),
        "穗行数": ("穗行数", "行"),
        "行粒数": ("行粒数", "粒"),
        "百粒重": ("百粒重", "g"),
        "穗型": ("果穗", ""), 
        "轴色": ("穗轴", ""),
        "粒型": ("粒型", ""),
        "粒色": ("粒色", ""),
        "折亩产量": ("折亩产量", "kg"),
        "邻ck亩产量": ("相邻ck亩产量", "kg"),
        "比邻ck±%": ("比邻增产", "%"),
    }

    # 打印结果
    results = {}
    for idx, row in result.iterrows():
        parts = []
        for col in output_order:
            if col in result.columns:
                val = row[col]
                label, unit = col_config.get(col, (col, ""))
                
                if col == "比对照±天":
                    try:
                        v = float(val)
                        v_abs = abs(v)
                        # 如果是整数，不显示小数位
                        v_str = f"{int(v_abs)}" if v_abs.is_integer() else f"{v_abs:.1f}"
                        if v < 0: parts.append(f"{label}早{v_str}{unit}")
                        elif v > 0: parts.append(f"{label}晚{v_str}{unit}")
                        else: parts.append(f"{label}持平")
                    except: parts.append(f"{label}{val}{unit}")
                elif col == "比邻ck±%":
                    try:
                        v = float(val)
                        v_str = f"{abs(v):.1f}"
                        if v < 0: parts.append(f"比邻减产{v_str}{unit}")
                        else: parts.append(f"比邻增产{v_str}{unit}")
                    except: parts.append(f"{label}{val}{unit}")
                elif col == "穗型":
                    s_val = str(val)
                    if not s_val.endswith("型"): s_val += "型"
                    parts.append(f"{label}{s_val}{unit}")
                elif unit == "级" and isinstance(val, (int, float)):
                    parts.append(f"{label}{val:.0f}{unit}")
                elif isinstance(val, (int, float)):
                    parts.append(f"{label}{val:.1f}{unit}")
                else:
                    parts.append(f"{label}{val}{unit}")
        
        final_str = f"{row[group_col]}：该品种{'，'.join(parts)}。"
        if row[group_col] in yield_stats:
            final_str += yield_stats[row[group_col]] + "。"
            
        results[row[group_col]] = final_str
    
    return results

def get_conclusions_dict(file_path: str) -> dict:
    """
    从文本文件中提取鉴定结论
    """
    if not os.path.exists(file_path):
        # print(f"文件不存在: {file_path}")
        return {}
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有鉴定结论句子：从“鉴定结论：”开始，到第一个“。”结束
    matches = re.findall(r'(鉴定结论：[^。]*。)', content)
    
    conclusions = {}
    for sentence in matches:
        # 提取品种名称：鉴定结论：{NAME}[抗|感|中|高]
        # 变量后通常紧跟抗性描述（抗、感、中抗、高抗等）
        name_match = re.search(r'鉴定结论：\s*(.*?)(?=[抗感中高])', sentence)
        if name_match:
            name = name_match.group(1).strip()
            conclusions[name] = sentence
            
    return conclusions

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "ny.xlsx")
    txt_path = os.path.join(base_dir, "data", "ny.txt")
    
    stats_map = get_stats_dict(data_path)
    conclusions_map = get_conclusions_dict(txt_path)
    
    for name, stat_text in stats_map.items():
        full_text = stat_text
        if name in conclusions_map:
            full_text += "\n" + conclusions_map[name]
        print(full_text + "\n")