import numpy as np

def finite_diff_grad(fn, x, eps=1e-5):
    # Returns numerical gradient wrt x (same shape)
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_pos = fn(x)
        x[idx] = old - eps
        f_neg = fn(x)
        x[idx] = old
        grad[idx] = (f_pos - f_neg) / (2 * eps)
        it.iternext()
    return grad

def rel_error(a, b, eps=1e-12):
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b) + eps
    return num / den
