FROM python:3.8-slim

RUN apt-get update && apt-get install -y curl

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

EXPOSE 8000
