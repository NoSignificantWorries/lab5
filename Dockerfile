FROM python:3.12-bookworm

WORKDIR /home/app

RUN apt update
RUN apt install libgl1-mesa-glx

COPY requirements.txt .

RUN pip install -r requirements.txt

