import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from core import Sequential, Dense, ReLU, Dropout, CrossEntropyLoss, Adam, Dataset, DataLoader


def to_one_hot(y, num_classes):
    y = np.array(y, dtype=int)
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def accuracy(logits, y_true):
    pred_classes = np.argmax(logits, axis=1)
    return np.mean(pred_classes == y_true)


print("Загрузка данных...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
X = X.astype(np.float32) / 255.0
y = y.astype(int)

X, _, y, _ = train_test_split(X, y, train_size=5000, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_dataset = Dataset(X_train, y_train)
val_dataset = Dataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


def create_model(use_dropout=False):
    layers = [
        Dense(784, 256),
        ReLU(),
    ]
    if use_dropout:
        layers.append(Dropout(p=0.5))  # Dropout с вероятностью 0.5

    layers.extend([
        Dense(256, 128),
        ReLU(),
    ])
    if use_dropout:
        layers.append(Dropout(p=0.3))

    layers.append(Dense(128, 10))
    return Sequential(layers)

def train_model(model, name, weight_decay=0.0):
    print(f"\n{'=' * 60}")
    print(f"Обучение: {name}")
    print(f"{'=' * 60}")

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_params(), lr=0.001, weight_decay=weight_decay)

    epochs = 30
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()  # Включаем режим обучения (Dropout активен)

        for X_batch, y_batch in train_loader:
            y_batch_one_hot = to_one_hot(y_batch, 10)
            logits = model.forward(X_batch)
            loss = loss_fn.forward(logits, y_batch_one_hot)
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()  # Выключаем Dropout для оценки
        train_logits = model.forward(X_train)
        train_acc = accuracy(train_logits, y_train)

        val_logits = model.forward(X_val)
        val_acc = accuracy(val_logits, y_val)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 5 == 0:
            print(f"Эпоха {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return history


print("\n" + "=" * 60)
print("ЭКСПЕРИМЕНТ: Влияние Dropout и L2 на переобучение")
print("=" * 60)

model1 = create_model(use_dropout=False)
history1 = train_model(model1, "Без регуляризации", weight_decay=0.0)

model2 = create_model(use_dropout=True)
history2 = train_model(model2, "С Dropout", weight_decay=0.0)

model3 = create_model(use_dropout=False)
history3 = train_model(model3, "С L2 (weight_decay=0.01)", weight_decay=0.01)


print("\nИТОГОВЫЕ РЕЗУЛЬТАТЫ (последняя эпоха)")
print("-" * 60)
print(f"{'Метод':<30} | {'Train Acc':<12} | {'Val Acc':<12} | {'Gap (переобучение)':<20}")
print("-" * 80)

for name, history in [
    ("Без регуляризации", history1),
    ("С Dropout", history2),
    ("С L2 (weight_decay)", history3)
]:
    train_acc = history['train_acc'][-1]
    val_acc = history['val_acc'][-1]
    gap = train_acc - val_acc
    print(f"{name:<30} | {train_acc:<12.4f} | {val_acc:<12.4f} | {gap:<20.4f}")
