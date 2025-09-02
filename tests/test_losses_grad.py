import numpy as np
from src.core.losses import softmax_cross_entropy, softmax_cross_entropy_backward
from tests.test_grad_check_numeric import finite_diff_grad, rel_error

def test_softmax_ce_backward_matches_numeric():
    rng = np.random.default_rng(1)
    N, C = 5, 4
    logits = rng.normal(size=(N, C)).astype(np.float64)
    y = rng.integers(0, C, size=(N,))
    y_one = np.zeros((N, C), dtype=np.float64)
    y_one[np.arange(N), y] = 1.0

    def f(L):
        return softmax_cross_entropy(L, y_one)

    g_num = finite_diff_grad(f, logits.copy(), eps=1e-6)
    g_ana = softmax_cross_entropy_backward(logits.copy(), y_one)
    err = rel_error(g_num, g_ana)
    assert err < 5e-6
