from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

# Загрузка данных
data = pd.read_csv('dataset.csv')
# print(data.head(5))

# Flask приложение
app = Flask(__name__)


@app.route('/')
def home():
    num_features = 32  # Количество ожидаемых признаков
    return render_template('index.html', num_features=num_features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных от пользователя
        user_input = []
        for i in range(32):
            value = request.form.get(f'feature{i + 1}')
            if value is None or not value.strip():  # Проверка на пустые значения
                raise ValueError(f"Признак feature{i + 1} отсутствует или пуст.")
            try:
                user_input.append(float(value))
            except ValueError:
                raise ValueError(f"Признак feature{i + 1} должен быть числом.")

        # Убедитесь, что входные данные имеют правильную форму
        input_array = np.array([user_input])  # Преобразование в 2D массив

        # Предсказание
        prediction = model.predict(input_array)  # Ожидается, что модель возвращает массив
        prediction_rescaled = prediction[0]
        print(prediction_rescaled)

        # Ограничение диапазона (0-100)
        prediction_rescale = max(0, min(0, prediction_rescaled))

        return render_template('index.html',
                               prediction_text=f'Предсказанная оценка: {prediction_rescale:.2f}'
                                               f'{prediction_rescaled:.2f}'
                                               f'{prediction:.2f}')
    except ValueError as ve:
        return render_template('index.html', prediction_text=f'Ошибка ввода: {str(ve)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Ошибка: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
