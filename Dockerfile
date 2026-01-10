FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

# For deployment on Render.com
CMD ["gunicorn","app:app","--host","0.0.0.0","--port","10000"]
