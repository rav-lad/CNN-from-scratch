# tests/test_utils_im2col.py
import numpy as np
from src.core.utils import im2col, col2im

def test_im2col_col2im_adjoint_property():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 5, 5)).astype(np.float64)
    KH, KW, stride, pad = 3, 3, 1, 1

    cols = im2col(x, (KH, KW), stride=stride, pad=pad).astype(np.float64)
    # random "columns" of matching shape
    c = rng.normal(size=cols.shape).astype(np.float64)

    left = np.sum(cols * c)

    xr = col2im(c, x.shape, (KH, KW), stride=stride, pad=pad).astype(np.float64)
    right = np.sum(x * xr)

    assert np.allclose(left, right, rtol=1e-10, atol=1e-10)
