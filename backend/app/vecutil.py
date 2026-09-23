"""Tiện ích vector. Mọi chỗ cần L2-normalize đều gọi hàm ở đây, không tự viết lại."""

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 theo từng hàng, trả về float32.

    Hàng có norm 0 được giữ nguyên là vector 0 thay vì thành NaN — một vector 0
    lọt vào index sẽ làm mọi truy vấn sau đó trả NaN và rất khó truy nguyên.

    :raises ValueError: nếu đầu vào không phải ma trận 2 chiều.
    """
    if matrix.ndim != 2:
        raise ValueError(f"cần ma trận 2 chiều, nhận được ndim={matrix.ndim}")
    out = matrix.astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, norms, out=out, where=norms > 0)
    return out
