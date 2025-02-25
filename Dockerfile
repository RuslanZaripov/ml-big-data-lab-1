FROM python:3.8-slim

RUN apt update && apt -y install locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt