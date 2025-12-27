"""Tests for pivot export helpers."""

import pandas as pd

from analysis.exports.pivot import nested_pivot_to_excel_bytes
from analysis.views.pivot_utils import build_nested_pivot_data


def test_nested_pivot_excel_bytes() -> None:
    """Ensure nested pivot export returns non-empty Excel bytes."""
    df = pd.DataFrame(
        {
            "Visit": ["V1", "V1", "V2", "V2"],
            "Group": ["A", "B", "A", "B"],
            "Value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    data = build_nested_pivot_data(
        df=df,
        row_key_cols=["Visit"],
        col_key_cols=["Group"],
        value_cols=["Value"],
        agg_names=["Mean - 平均值"],
    )
    output = nested_pivot_to_excel_bytes(data)
    assert isinstance(output, (bytes, bytearray))
    assert len(output) > 200
