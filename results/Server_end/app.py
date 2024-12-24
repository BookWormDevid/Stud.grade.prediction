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
    try:
        # Получение значений
        features = [
            float(request.form['gender']),
            float(request.form['age']),
            float(request.form['address']),
            float(request.form['famsize']),
            float(request.form['Pstatus']),
            float(request.form['Medu']),
            float(request.form['Fedu']),
            float(request.form['Mjob']),
            float(request.form['Fjob']),
            float(request.form['reason']),
            float(request.form['guardian']),
            float(request.form['traveltime']),
            float(request.form['studytime']),
            float(request.form['failures']),
            float(request.form['schoolsup']),
            float(request.form['famsup']),
            float(request.form['paid']),
            float(request.form['activities']),
            float(request.form['nursery']),
            float(request.form['higher']),
            float(request.form['internet']),
            float(request.form['romantic']),
            float(request.form['famrel']),
            float(request.form['freetime']),
            float(request.form['goout']),
            float(request.form['Dalc']),
            float(request.form['Walc']),
            float(request.form['health']),
            float(request.form['absences']),
            float(request.form['G1']),
            float(request.form['G2']),
        ]
        input_data = np.array([features])

        # Нормализация данных
        input_data_normalized = (input_data - x_min) / (x_max - x_min)
        prediction = model.predict(input_data_normalized)
        prediction = prediction.flatten()
        return render_template('result.html', prediction=f"{prediction[0], '+-12%':.2f}")
    except KeyError as e:
        return f"Missing field in request: {e}", 400
    except Exception as e:
        return f"An error occurred: {e}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
