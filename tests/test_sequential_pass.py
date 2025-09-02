import numpy as np
from src.models.convnet_small import lenet_mnist
from src.core.losses import softmax_cross_entropy, softmax_cross_entropy_backward
from src.core.utils import one_hot

def test_lenet_mnist_forward_backward_shapes():
    model = lenet_mnist(num_classes=10)
    x = np.random.randn(8, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, size=(8,))
    logits = model.forward(x, training=True)
    assert logits.shape == (8, 10)
    y1 = one_hot(y, 10)
    loss = softmax_cross_entropy(logits, y1)
    grad_logits = softmax_cross_entropy_backward(logits, y1)
    dx = model.backward(grad_logits)
    assert dx.shape == x.shape
