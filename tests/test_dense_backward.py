import numpy as np
from src.layers.dense import Dense
from tests.test_grad_check_numeric import finite_diff_grad, rel_error

def test_dense_backward_input_and_params():
    rng = np.random.default_rng(2)
    N, in_f, out_f = 4, 6, 3
    x = rng.normal(size=(N, in_f)).astype(np.float64)
    layer = Dense(in_f, out_f, bias=True, weight_init="xavier_uniform", rng=rng)

    # forward
    y = layer.forward(x)
    # simple scalar loss: sum of outputs
    grad_out = np.ones_like(y, dtype=np.float64)
    dx = layer.backward(grad_out)

    # check dX numeric
    def f_input(xx):
        xx = xx.reshape(N, in_f)
        return np.sum(layer.forward(xx))

    dx_num = finite_diff_grad(f_input, x.copy(), eps=1e-6)
    assert rel_error(dx, dx_num) < 1e-6

    # check dW numeric
    W = layer.W.astype(np.float64)
    def f_W(Wvar):
        Wbak = layer.W.copy()
        layer.W[...] = Wvar
        val = np.sum(layer.forward(x))
        layer.W[...] = Wbak
        return val

    dW_num = finite_diff_grad(f_W, W.copy(), eps=1e-6)
    assert rel_error(layer.grads()["W"], dW_num) < 1e-6

    # check db numeric
    b = layer.b.astype(np.float64)
    def f_b(bvar):
        bbak = layer.b.copy()
        layer.b[...] = bvar
        val = np.sum(layer.forward(x))
        layer.b[...] = bbak
        return val

    db_num = finite_diff_grad(f_b, b.copy(), eps=1e-6)
    assert rel_error(layer.grads()["b"], db_num) < 1e-6
