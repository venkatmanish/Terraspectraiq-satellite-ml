import numpy as np
import pytest

from terraspectra import schema
from terraspectra.csv_guard import CSVValidationError, read_csv_bytes, validate


def to_bytes(frame):
    return frame.to_csv(index=False).encode()


def test_official_test_csv_is_valid(test_csv):
    up = validate(read_csv_bytes(to_bytes(test_csv)))
    assert up.X.shape == (2000, 36)
    assert list(up.X.columns) == list(schema.FEATURES)
    assert set(up.y) == set(schema.CLASS_LABELS)
    assert up.warnings == []


def test_reordered_and_extra_columns_are_aligned(test_csv):
    shuffled = test_csv.sample(frac=1.0, axis=1, random_state=0).assign(notes="x")
    up = validate(read_csv_bytes(to_bytes(shuffled)))
    assert list(up.X.columns) == list(schema.FEATURES)
    np.testing.assert_array_equal(up.X.to_numpy(), test_csv[list(schema.FEATURES)].to_numpy())
    assert any("reordered" in w for w in up.warnings)
    assert any("extra column" in w for w in up.warnings)


def test_missing_feature_is_rejected(test_csv):
    with pytest.raises(CSVValidationError, match="missing"):
        validate(test_csv.drop(columns=["r2c2_b3"]))


def test_missing_target_rejected_when_required(test_csv):
    with pytest.raises(CSVValidationError, match="required"):
        validate(test_csv.drop(columns=[schema.TARGET]), require_target=True)


def test_missing_target_allowed_for_prediction_only(test_csv):
    up = validate(test_csv.drop(columns=[schema.TARGET]))
    assert up.y is None


def test_class_six_is_rejected(test_csv):
    bad = test_csv.head(10).copy()
    bad.loc[0, schema.TARGET] = 6
    with pytest.raises(CSVValidationError, match="class 6") as err:
        validate(bad)
    assert "Unknown class code(s) [6]." in str(err.value)  # plain ints, no numpy repr


def test_non_numeric_feature_is_rejected(test_csv):
    bad = test_csv.head(5).astype({"r1c1_b1": object})
    bad.loc[2, "r1c1_b1"] = "bright"
    with pytest.raises(CSVValidationError, match="non-numeric"):
        validate(bad)


def test_missing_value_is_rejected_not_imputed(test_csv):
    bad = test_csv.head(5).astype({"r1c1_b1": float})
    bad.loc[1, "r1c1_b1"] = np.nan
    with pytest.raises(CSVValidationError, match="not imputed"):
        validate(bad)


def test_out_of_range_value_is_rejected(test_csv):
    bad = test_csv.head(5).copy()
    bad.loc[0, "r3c3_b4"] = 300
    with pytest.raises(CSVValidationError, match="0–255"):
        validate(bad)


def test_empty_and_malformed_files():
    with pytest.raises(CSVValidationError, match="empty"):
        read_csv_bytes(b"")
    with pytest.raises(CSVValidationError, match="no data rows"):
        read_csv_bytes(b"a,b,c\n")
    with pytest.raises(CSVValidationError):
        read_csv_bytes(b'a,b\n1,"2\n3,4,5,6\n')


def test_single_class_file_is_accepted_with_warning(test_csv):
    one = test_csv[test_csv[schema.TARGET] == 2].head(20)
    up = validate(one)
    assert set(up.y) == {2}
    assert any("absent" in w for w in up.warnings)
