import logging
from os import mkdir
from os.path import exists
from logging.handlers import RotatingFileHandler
from aiogram import Bot, Dispatcher
from settings import settings
from handlers import router


# создание папки для хранения логов
if not exists("logs"):
    mkdir("logs")

# конфигурация параметров логирования
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            filename=settings.LOG_FILE,
            mode="a",
            maxBytes=500000,
            backupCount=5,
            delay=True,
        )
    ],
    level=logging.NOTSET,
)


dp = Dispatcher()
dp.include_router(router)
