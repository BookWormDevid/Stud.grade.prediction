import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

# Загрузка данных
data = pd.read_csv('dataset.csv')
print(data.head(5))

dataset = pd.DataFrame(data)
dataset.head(5)

# Целевая переменная
target = pd.DataFrame(dataset[["G1", "G2", "G3"]], columns=["G3"])
dataset = shuffle(dataset)
dataset.head(5)

# Убираем столбец G3 из признаков
features = dataset.drop(columns=["G3"])

# Преобразуем данные в массивы
x = np.array(features)
y = np.array(target)

# Нормализация данных
x_min = np.min(x, axis=0)
x_max = np.max(x, axis=0)
x_normalized = (x - x_min) / (x_max - x_min)

y_min = np.min(y)
y_max = np.max(y)
y_normalized = (y - y_min) / (y_max - y_min)

# Разделение на обучающую и тестовую выборки
train_size = int(0.8 * x_normalized.shape[0])
x_train, x_test = x_normalized[:train_size], x_normalized[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='elu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='relu')
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.MeanSquaredLogarithmicError())

# Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=16)

# Сохранение модели
model.save("model.h5")

# Предсказания
predictions = model.predict(x_test)
# Локальная проверка
test_sample = x_test[0].reshape(1, -1)
predicted_value = model.predict(test_sample)
print(f"Predicted value: {predicted_value[0][0]:.2f}")

# Нормализация
test_sample_normalized = (x_test[0] - x_min) / (x_max - x_min)
predicted_value = model.predict(test_sample_normalized.reshape(1, -1))

# Де-нормализация
predicted_value = predicted_value * (y_max - y_min) + y_min
print(f"Predicted value (denormalized): {predicted_value[0][0]:.2f}")
