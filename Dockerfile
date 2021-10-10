FROM python:3.9.0-buster

WORKDIR /code
COPY requirements.txt /code/requirements.txt
COPY requirements-dev.txt /code/requirements-dev.txt

RUN pip install -r requirements-dev.txt