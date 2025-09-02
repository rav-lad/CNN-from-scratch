import numpy as np
from src.layers.conv2d import Conv2D
from tests.test_grad_check_numeric import finite_diff_grad, rel_error

def test_conv2d_backward_input_and_params():
    rng = np.random.default_rng(3)
    N, Cin, H, W = 2, 2, 6, 6
    Cout, KH, KW, stride, pad = 2, 3, 3, 1, 1
    x = rng.normal(size=(N, Cin, H, W)).astype(np.float64)
    layer = Conv2D(Cin, Cout, (KH, KW), stride=stride, padding=pad, bias=True, rng=rng)

    y = layer.forward(x)
    grad_out = np.ones_like(y, dtype=np.float64)
    dx = layer.backward(grad_out)

    # numeric dX
    def f_input(xx):
        xx = xx.reshape(N, Cin, H, W)
        return np.sum(layer.forward(xx))
    dx_num = finite_diff_grad(f_input, x.copy(), eps=1e-5)
    assert rel_error(dx, dx_num) < 3e-5

    # numeric dW
    W = layer.W.astype(np.float64)
    def f_W(Wvar):
        Wbak = layer.W.copy()
        layer.W[...] = Wvar
        val = np.sum(layer.forward(x))
        layer.W[...] = Wbak
        return val
    dW_num = finite_diff_grad(f_W, W.copy(), eps=1e-5)
    assert rel_error(layer.grads()["W"], dW_num) < 5e-5

    # numeric db
    b = layer.b.astype(np.float64)
    def f_b(bvar):
        bbak = layer.b.copy()
        layer.b[...] = bvar
        val = np.sum(layer.forward(x))
        layer.b[...] = bbak
        return val
    db_num = finite_diff_grad(f_b, b.copy(), eps=1e-5)
    assert rel_error(layer.grads()["b"], db_num) < 5e-6
