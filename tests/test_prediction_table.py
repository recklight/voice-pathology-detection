"""The prediction table that ``predict`` writes."""

from __future__ import annotations

import numpy as np
import pytest

from voice_pathology_detection.prediction_table import (
    FILE_COLUMN,
    PREDICTION_COLUMN,
    build_prediction_table,
    read_prediction_table,
    write_prediction_table,
)


def test_the_table_has_the_two_expected_columns() -> None:
    table = build_prediction_table(["001.wav", "002.wav"], [0, 3])
    assert list(table.columns) == [FILE_COLUMN, PREDICTION_COLUMN]
    assert table[FILE_COLUMN].tolist() == ["001", "002"]
    assert table[PREDICTION_COLUMN].tolist() == [0, 3]


def test_identifiers_keep_their_leading_zeros_through_a_round_trip(tmp_path) -> None:
    path = write_prediction_table(tmp_path / "pred.csv", ["001.wav", "010.wav"], [1, 2])
    restored = read_prediction_table(path)
    assert restored[FILE_COLUMN].tolist() == ["001", "010"]
    assert restored[PREDICTION_COLUMN].tolist() == [1, 2]


def test_the_suffix_can_be_kept() -> None:
    table = build_prediction_table(["001.wav"], [0], strip_suffix=False)
    assert table[FILE_COLUMN].tolist() == ["001.wav"]


def test_a_spreadsheet_round_trip_preserves_the_predictions(tmp_path) -> None:
    pytest.importorskip("openpyxl")
    path = write_prediction_table(tmp_path / "pred.xlsx", ["001", "002", "003"], [0, 1, 2])
    restored = read_prediction_table(path)
    assert restored[PREDICTION_COLUMN].tolist() == [0, 1, 2]
    assert restored[FILE_COLUMN].tolist() == ["001", "002", "003"]


def test_the_table_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="exactly one prediction"):
        build_prediction_table(["001", "002"], [0])
    with pytest.raises(ValueError, match="empty prediction table"):
        build_prediction_table([], [])
    with pytest.raises(ValueError, match="outside the valid range"):
        build_prediction_table(["001"], [7])
    with pytest.raises(ValueError, match="more than once"):
        build_prediction_table(["001.wav", "001.mp3"], [0, 1])


def test_reading_reports_what_is_wrong(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no such prediction table"):
        read_prediction_table(tmp_path / "absent.csv")

    broken = tmp_path / "broken.csv"
    broken.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the column"):
        read_prediction_table(broken)


def test_a_numpy_prediction_array_is_accepted(tmp_path) -> None:
    path = write_prediction_table(tmp_path / "a.csv", ["001", "002"], np.array([2, 3]))
    assert read_prediction_table(path)[PREDICTION_COLUMN].tolist() == [2, 3]
