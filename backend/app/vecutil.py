"""Vector utilities. Every place that needs L2-normalize calls the function here instead of rewriting it."""

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize row by row, returning float32.

    A row with norm 0 is kept as a zero vector instead of becoming NaN — a
    zero vector that slips into the index would make every subsequent query
    return NaN, which is very hard to trace back.

    :raises ValueError: if the input isn't a 2-dimensional matrix.
    """
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-dimensional matrix, got ndim={matrix.ndim}")
    out = matrix.astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, norms, out=out, where=norms > 0)
    return out
