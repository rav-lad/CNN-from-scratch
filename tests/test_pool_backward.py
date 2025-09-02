import numpy as np
from src.layers.pooling import MaxPool2D, AvgPool2D
from tests.test_grad_check_numeric import finite_diff_grad, rel_error

def test_maxpool2d_backward_numeric():
    rng = np.random.default_rng(4)
    # Avoid ties by adding tiny noise
    x = rng.normal(size=(2, 3, 5, 5)).astype(np.float64) + 1e-3 * np.arange(2*3*5*5).reshape(2,3,5,5)
    pool = MaxPool2D(kernel_size=2, stride=2)
    y = pool.forward(x)
    grad_out = np.ones_like(y, dtype=np.float64)
    dx = pool.backward(grad_out)

    def f_input(xx):
        xx = xx.reshape(2,3,5,5)
        return np.sum(pool.forward(xx))
    dx_num = finite_diff_grad(f_input, x.copy(), eps=1e-6)
    assert rel_error(dx, dx_num) < 2e-6

def test_avgpool2d_backward_numeric():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3, 6, 6)).astype(np.float64)
    pool = AvgPool2D(kernel_size=2, stride=2)
    y = pool.forward(x)
    grad_out = np.ones_like(y, dtype=np.float64)
    dx = pool.backward(grad_out)

    def f_input(xx):
        xx = xx.reshape(2,3,6,6)
        return np.sum(pool.forward(xx))
    dx_num = finite_diff_grad(f_input, x.copy(), eps=1e-6)
    assert rel_error(dx, dx_num) < 2e-6
