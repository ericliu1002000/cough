"""Shared constants used across the app."""

SUBJECT_ID_ALIASES = [
    
    "USUBJID",
    "SUBJECTID",
    "SUBJID",
    "patient_id",
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
