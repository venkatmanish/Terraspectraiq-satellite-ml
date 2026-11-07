import numpy as np
import pandas as pd
import pytest

from terraspectra import schema
from terraspectra.data import DataIntegrityError, load_partition, neighbourhood, overlap_rate


def test_partitions_match_published_sizes():
    train, test = load_partition("train"), load_partition("test")
    assert train.shape == (4435, 37)
    assert test.shape == (2000, 37)


def test_schema_has_36_pixel_major_features():
    assert len(schema.FEATURES) == 36
    assert schema.FEATURES[:4] == ("r1c1_b1", "r1c1_b2", "r1c1_b3", "r1c1_b4")
    # UCI attributes 17-20 (1-based) are the centre pixel
    assert schema.FEATURES[16:20] == schema.CENTRE_FEATURES


def test_class_codes_are_original_and_skip_six():
    assert schema.CLASS_LABELS == (1, 2, 3, 4, 5, 7)
    for split in ("train", "test"):
        labels = set(load_partition(split)[schema.TARGET].unique())
        assert labels == {1, 2, 3, 4, 5, 7}


def test_values_are_finite_8bit_integers():
    frame = load_partition("train")
    x = frame[list(schema.FEATURES)].to_numpy()
    assert np.issubdtype(x.dtype, np.integer)
    assert x.min() >= 0 and x.max() <= 255
    assert not frame.isna().any().any()


def test_first_row_matches_raw_file_order():
    first_line = (schema.RAW_DIR / "sat.trn").read_text().splitlines()[0].split()
    row = load_partition("train").iloc[0]
    assert row[list(schema.FEATURES)].tolist() == [int(v) for v in first_line[:36]]
    assert row[schema.TARGET] == int(first_line[36])


def test_neighbourhood_reshape_puts_attributes_17_to_20_in_centre():
    row = load_partition("train").iloc[0]
    grid = neighbourhood(row)
    assert grid.shape == (3, 3, 4)
    np.testing.assert_array_equal(grid[1, 1], row[list(schema.CENTRE_FEATURES)].to_numpy())


def test_checksum_mismatch_is_detected(tmp_path):
    for name in ("sat.trn", "sat.tst"):
        (tmp_path / name).write_text((schema.RAW_DIR / name).read_text().replace("92", "93", 1))
    with pytest.raises(DataIntegrityError):
        load_partition("train", raw_dir=tmp_path)


def test_overlap_detection_on_synthetic_shift():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(4, 4, 4))  # 4x4 pixels, 4 bands
    def window(r, c):
        return image[r:r + 3, c:c + 3].reshape(-1)
    a = pd.DataFrame([window(0, 0)], columns=list(schema.FEATURES))
    b = pd.DataFrame([window(0, 1)], columns=list(schema.FEATURES))   # one pixel to the right
    far = pd.DataFrame([rng.integers(0, 255, 36)], columns=list(schema.FEATURES))
    assert overlap_rate(a, b) == 1.0
    assert overlap_rate(a, far) == 0.0
    assert overlap_rate(a, a, exclude_self=True) == 0.0
