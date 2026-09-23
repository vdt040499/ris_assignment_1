import numpy as np
import pytest

from app.vecutil import l2_normalize


def test_rows_become_unit_length():
    matrix = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    out = l2_normalize(matrix)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_direction_is_preserved():
    matrix = np.array([[3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(l2_normalize(matrix), [[0.6, 0.8]], atol=1e-6)


def test_zero_row_stays_zero_instead_of_nan():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(matrix)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[0], [0.0, 0.0])


def test_output_is_float32_regardless_of_input_dtype():
    assert l2_normalize(np.array([[1.0, 1.0]], dtype=np.float64)).dtype == np.float32


def test_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        l2_normalize(np.array([1.0, 2.0], dtype=np.float32))
