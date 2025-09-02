import numpy as np
from src.core.regularizers import max_norm

def test_max_norm_inplace():
    W = np.array([[3.0, 4.0]], dtype=np.float32)  # norm=5
    params = {"W": W}
    max_norm(params, max_value=2.0, exclude=())
    # Must be scaled to norm 2, same object updated
    assert np.isclose(np.linalg.norm(W), 2.0, atol=1e-6)
