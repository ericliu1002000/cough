"""Tests for calculation graph filter operators."""

import pandas as pd
import pytest

from analysis.services.calculation_graph import _apply_filter


def test_filter_in_excludes_values() -> None:
    """Rows matching IN values are removed."""
    df = pd.DataFrame({"col": ["A", "B", "C", "D"]})
    result = _apply_filter(
        df,
        {"field": "col", "op": "in", "values": ["B", "D"]},
    )
    assert result["col"].tolist() == ["A", "C"]


def test_filter_not_in_excludes_non_list_values() -> None:
    """Rows matching NOT IN condition are removed."""
    df = pd.DataFrame({"col": ["A", "B", "C", "D"]})
    result = _apply_filter(
        df,
        {"field": "col", "op": "not_in", "values": ["B", "D"]},
    )
    assert result["col"].tolist() == ["A", "C"]


@pytest.mark.parametrize(
    ("op", "target", "expected"),
    [
        ("gt", 0.3, [0.1, 0.3]),
        ("ge", 0.3, [0.1]),
        ("lt", 0.3, [0.3, 0.5]),
        ("le", 0.3, [0.5]),
        ("eq", 0.3, [0.1, 0.5]),
        ("ne", 0.3, [0.1, 0.5]),
    ],
)
def test_filter_numeric_ops_exclude_matches(
    op: str, target: float, expected: list[float]
) -> None:
    """Numeric operators remove matching rows."""
    df = pd.DataFrame({"num": [0.1, 0.3, 0.5]})
    result = _apply_filter(
        df,
        {"field": "num", "op": op, "values": [str(target)]},
    )
    assert result["num"].tolist() == expected


def test_filter_is_null_excludes_null_like() -> None:
    """NULL/empty values are removed for is_null."""
    df = pd.DataFrame({"col": [None, "", "foo"]})
    result = _apply_filter(df, {"field": "col", "op": "is_null", "values": []})
    assert result["col"].tolist() == ["foo"]


def test_filter_is_not_null_keeps_non_null() -> None:
    """Non-null values are kept for is_not_null."""
    df = pd.DataFrame({"col": [None, "", "foo"]})
    result = _apply_filter(
        df, {"field": "col", "op": "is_not_null", "values": []}
    )
    assert result["col"].tolist() == ["foo"]


def test_filter_like_excludes_matches() -> None:
    """LIKE pattern matches are removed (SQL-style)."""
    df = pd.DataFrame({"col": ["alpha", "beta", "alphabet", "ALPHA"]})
    result = _apply_filter(
        df,
        {"field": "col", "op": "like", "values": ["%alpha%"]},
    )
    assert result["col"].tolist() == ["beta", "ALPHA"]
