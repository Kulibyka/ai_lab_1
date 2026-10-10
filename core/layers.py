import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.grad_W = None
        self.grad_b = None
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        self.grad_W = np.dot(self.input.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_params(self):
        return [layer for layer in self.layers if hasattr(layer, 'W')]

    def train(self):
        """Включает режим обучения (для Dropout)"""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        """Включает режим тестирования (для Dropout)"""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False


class Dropout:
    """Dropout слой. Во время обучения случайно обнуляет нейроны с вероятностью p."""
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True  # Флаг режима (train/test)

    def forward(self, x):
        if not self.training:
            return x

        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        return x * self.mask / (1 - self.p)

    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask / (1 - self.p)