from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

# Загрузка данных
data = pd.read_csv('dataset.csv')

# Flask приложение
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из формы
    features = [float(request.form[f'feature{i}']) for i in range(1, 33)]
    # Преобразование данных в массив NumPy
    input_data = np.array([features])
    # Предсказание модели
    prediction = model.predict(input_data)
    return f"Predicted value: {prediction[0][0]:.2f}"


if __name__ == "__main__":
    app.run(debug=True)
