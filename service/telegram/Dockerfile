FROM python:3.12

RUN mkdir /telegram

COPY . /telegram

WORKDIR /telegram

RUN pip install -r requirements.txt --no-cache-dir

CMD ["python", "main.py"]
