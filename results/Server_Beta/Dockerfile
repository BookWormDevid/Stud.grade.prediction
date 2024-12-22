# Используем базовый образ Python
FROM python:3.9-slim

# Указываем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё содержимое текущей директории внутрь контейнера
COPY . /app

# Указываем команду для запуска приложения
CMD ["python", "app.py"]
