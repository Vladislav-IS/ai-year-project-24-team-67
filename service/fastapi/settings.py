from pydantic import Field
from pydantic_settings import BaseSettings

import sqlite3


class Settings(BaseSettings):
    # директория, в которую сохраняются модели
    MODEL_DIR: str = Field(default='trash/models')
     
    # количество CPU (приходится ставить побольше для асинхронности)
    # видимо, я не совсем верно реализовал подсчет активных процессов
    NUM_CPUS: int = Field(default=6) 

    # максимальное количество моделей в инференсе
    NUM_MODELS: int = Field(default=2)


settings = Settings()