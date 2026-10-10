import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSELoss(Loss):
    """Среднеквадратичная ошибка (для регрессии)"""

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        batch_size = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / batch_size


class CrossEntropyLoss(Loss):
    """Кросс-энтропия (для классификации)."""

    def __init__(self):
        self.probs = None
        self.y_true = None
        self.batch_size = None

    def forward(self, logits, y_true):
        self.y_true = y_true
        self.batch_size = logits.shape[0]

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        eps = 1e-15
        probs_clipped = np.clip(self.probs, eps, 1.0 - eps)
        loss = -np.sum(y_true * np.log(probs_clipped)) / self.batch_size
        return loss

    def backward(self):
        return (self.probs - self.y_true) / self.batch_size