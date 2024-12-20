from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf

# Загрузка модели
model = tf.keras.models.load_model("model.h5")

# Загрузка данных
data = pd.read_csv('dataset.csv')

x_min = data.iloc[:, :-1].min().values  # Исключаем целевую переменную
x_max = data.iloc[:, :-1].max().values
y_min = data.iloc[:, -1].min()  # Целевая переменная
y_max = data.iloc[:, -1].max()

# Flask приложение
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Чтение данных из формы
    features = [float(request.form[f'feature{i}']) for i in range(1, 32)]
    input_data = np.array([features])

    # Нормализация данных
    input_data_normalized = (input_data - x_min) / (x_max - x_min)

    prediction = model.predict(input_data_normalized)
    prediction1 = prediction.flatten()
    return f"Predicted value normalized: {prediction1[0]:.2f}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
