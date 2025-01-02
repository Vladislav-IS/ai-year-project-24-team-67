FROM python:3.12

RUN mkdir /fastapi

COPY . /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "log_config.json", "--reload"]
