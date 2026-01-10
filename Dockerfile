FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .

RUN apt update -y \
    && apt install -y awscli \
    && pip install --no-cache-dir -r requirements.txt \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

EXPOSE 10000

CMD ["gunicorn","app:app","--bind","0.0.0.0:10000"]
