import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import callbacks
import pickle

data = pd.read_csv('dataset.csv')
print(data.head(5))

dataset = pd.DataFrame(data)
dataset.head(5)

target = pd.DataFrame(dataset[["G1", "G2", "G3"]], columns=["G3"])
dataset = shuffle(dataset)
dataset.head(5)

features = dataset

x = np.array(features)
y = np.array(target)

x_min = np.min(x, axis=0)
x_max = np.max(x, axis=0)

x_normalized = (x - x_min) / (x_max - x_min)

y_min = np.min(y)
y_max = np.max(y)

y_normalized = (y - y_min) / (y_max - y_min)

train_size = int(0.8 * x_normalized.shape[0])

x_train, x_test = x_normalized[:train_size], x_normalized[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1], )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='leaky_relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='leaky_relu'),
    tf.keras.layers.Dense(1, activation='relu')
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="MSLE")

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=16)

test_loss = model.evaluate(x_test, y_test)
print(f'Test loss (MSE): {test_loss}')
predictions = model.predict(x_test)


avg_error = []

for i in range(100):
    print(f'Предсказанная оценка: {round(predictions[i][0]):.2f}, Реальная оценка : {y_test[i][0]}')
    avg_error.append(abs(predictions[i][0]) - y_test[i][0])
print(sum(avg_error)/len(avg_error))

# Построение графиков точности и потерь

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'b', label='Потери на обучении')
plt.plot(epochs, val_loss, 'r', label='Потери на тестах')
plt.title('Потери обучения и тестов')
plt.legend()

plt.show()

pickle.dump(model, open("model.pkl", "wb"))
