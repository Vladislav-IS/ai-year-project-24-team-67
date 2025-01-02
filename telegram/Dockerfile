FROM python:3.12

WORKDIR /app

COPY requirments.txt requirments.txt

RUN pip install -r requirments.txt

COPY . .

CMD ["python", "bot.py"]
