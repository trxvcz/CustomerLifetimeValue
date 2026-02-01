FROM python:3.9-slim
LABEL authors="pawel"
WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY models/RandomForestRegressor.joblib .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]