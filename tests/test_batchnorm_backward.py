import numpy as np
from src.layers.batchnorm import BatchNorm2D
from tests.test_grad_check_numeric import finite_diff_grad, rel_error

def test_batchnorm2d_backward_numeric():
    rng = np.random.default_rng(6)
    N, C, H, W = 3, 4, 5, 5
    x = rng.normal(size=(N, C, H, W)).astype(np.float64)
    bn = BatchNorm2D(C, eps=1e-5, momentum=0.0)
    _ = bn.forward(x, training=True)

    # constant upstream grad -> true dx should be exactly 0 (sum of normalized = 0)
    grad_out = np.ones_like(x, dtype=np.float64)
    dx = bn.backward(grad_out)

    def f_input(xx):
        xx = xx.reshape(N, C, H, W)
        # forward recomputes mean/var every call in training mode
        return np.sum(bn.forward(xx, training=True))

    dx_num = finite_diff_grad(f_input, x.copy(), eps=1e-6)

    # Quand la norme vraie est ~0, la comparaison relative est ill- posée.
    # On vérifie l'erreur absolue.
    err_abs = np.max(np.abs(dx - dx_num))
    assert err_abs < 1e-8

    # Et on garde des checks solides pour dgamma/dbeta (non nuls en général)
    # Recalcule forward pour rafraîchir le cache
    _ = bn.forward(x, training=True)
    dgamma = bn.grads()["gamma"].copy()
    dbeta = bn.grads()["beta"].copy()

    # Numerical for gamma/beta
    gamma = bn.gamma.astype(np.float64)
    def f_gamma(gv):
        gb = bn.gamma.copy()
        bn.gamma[...] = gv
        val = np.sum(bn.forward(x, training=True))
        bn.gamma[...] = gb
        return val
    dgamma_num = finite_diff_grad(f_gamma, gamma.copy(), eps=1e-6)

    if np.linalg.norm(dgamma_num) < 1e-8:
        assert np.allclose(dgamma, dgamma_num, atol=1e-8)
    else:
        assert rel_error(dgamma, dgamma_num) < 1e-6

    beta = bn.beta.astype(np.float64)
    def f_beta(bv):
        bb = bn.beta.copy()
        bn.beta[...] = bv
        val = np.sum(bn.forward(x, training=True))
        bn.beta[...] = bb
        return val
    dbeta_num = finite_diff_grad(f_beta, beta.copy(), eps=1e-6)

    if np.linalg.norm(dbeta_num) < 1e-8:
        assert np.allclose(dbeta, dbeta_num, atol=1e-8)
    else:
        assert rel_error(dbeta, dbeta_num) < 1e-6