import numpy as np


class SGD:
    """Стохастический градиентный спуск с опциональной L2-регуляризацией"""

    def __init__(self, params, lr=0.01, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            grad_W = p.grad_W + self.weight_decay * p.W
            grad_b = p.grad_b + self.weight_decay * p.b

            p.W -= self.lr * grad_W
            p.b -= self.lr * grad_b

    def zero_grad(self):
        for p in self.params:
            p.grad_W = np.zeros_like(p.W)
            p.grad_b = np.zeros_like(p.b)


class Adam:
    """Оптимизатор Adam с опциональной L2-регуляризацией (Weight Decay)"""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

        for p in self.params:
            pid = id(p)
            self.m[pid] = {
                'W': np.zeros_like(p.W),
                'b': np.zeros_like(p.b)
            }
            self.v[pid] = {
                'W': np.zeros_like(p.W),
                'b': np.zeros_like(p.b)
            }

    def step(self):
        self.t += 1
        for p in self.params:
            pid = id(p)

            # L2 regularization: добавляем штраф к градиенту
            grad_W = p.grad_W + self.weight_decay * p.W
            grad_b = p.grad_b + self.weight_decay * p.b

            self.m[pid]['W'] = self.beta1 * self.m[pid]['W'] + (1 - self.beta1) * grad_W
            self.v[pid]['W'] = self.beta2 * self.v[pid]['W'] + (1 - self.beta2) * (grad_W ** 2)

            m_hat_w = self.m[pid]['W'] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[pid]['W'] / (1 - self.beta2 ** self.t)

            p.W -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)

            self.m[pid]['b'] = self.beta1 * self.m[pid]['b'] + (1 - self.beta1) * grad_b
            self.v[pid]['b'] = self.beta2 * self.v[pid]['b'] + (1 - self.beta2) * (grad_b ** 2)

            m_hat_b = self.m[pid]['b'] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v[pid]['b'] / (1 - self.beta2 ** self.t)

            p.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad_W = np.zeros_like(p.W)
            p.grad_b = np.zeros_like(p.b)