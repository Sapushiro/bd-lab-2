FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY config.ini .
COPY src ./src
COPY experiments/log_reg.sav ./experiments/log_reg.sav
COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]