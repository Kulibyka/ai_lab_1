import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from core import Sequential, Dense, ReLU, CrossEntropyLoss, Adam, Dataset, DataLoader


def to_one_hot(y, num_classes):
    y = np.array(y, dtype=int)
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def accuracy(logits, y_true):
    pred_classes = np.argmax(logits, axis=1)
    return np.mean(pred_classes == y_true)

print("Загрузка данных MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
X = X.astype(np.float32) / 255.0  # Нормализация [0, 1]
y = y.astype(int)

X, _, y, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = Dataset(X_train, y_train)
val_dataset = Dataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = Sequential([
    Dense(input_size=784, output_size=128),
    ReLU(),
    Dense(input_size=128, output_size=10)
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.get_params(), lr=0.005)
epochs = 20
print(f"\nНачало обучения на {epochs} эпохах...")

best_val_acc = 0.0
patience = 3  # Сколько эпох терпеть без улучшений
no_improve_count = 0

for epoch in range(1, epochs + 1):
    model.train_loss = 0
    batches = 0

    for X_batch, y_batch in train_loader:
        y_batch_one_hot = to_one_hot(y_batch, num_classes=10)

        logits = model.forward(X_batch)
        loss = loss_fn.forward(logits, y_batch_one_hot)
        model.train_loss += loss
        batches += 1

        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step()
        optimizer.zero_grad()

    val_logits = model.forward(X_val)
    val_acc = accuracy(val_logits, y_val)
    avg_loss = model.train_loss / batches

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_count = 0
        print(f"Новое лучшее качество: {val_acc:.4f}")

        # Сохраняем веса лучшей модели (для примера с 2 слоями Dense)
        np.savez('best_model.npz',
                 W1=model.layers[0].W, b1=model.layers[0].b,
                 W2=model.layers[2].W, b2=model.layers[2].b)
    else:
        no_improve_count += 1

    if no_improve_count >= patience:
        print(f"\nEarly stopping на эпохе {epoch}. Лучший Val Acc: {best_val_acc:.4f}")
        break

    print(f"Эпоха {epoch:2d} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

print("\nОбучение завершено успешно!")