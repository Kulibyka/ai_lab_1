import numpy as np


class Dataset:
    """Базовый класс для набора данных"""

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataLoader:
    """Итератор, который разбивает данные на батчи и перемешивает их"""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_idx = indices[start_idx: start_idx + self.batch_size]
            yield self.dataset.X[batch_idx], self.dataset.y[batch_idx]