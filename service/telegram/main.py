import asyncio
import logging
from logging.handlers import RotatingFileHandler
from os import mkdir
from os.path import exists

from aiogram import Dispatcher
from handlers import bot, router
from settings import settings

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
            encoding='utf-8'
        )
    ],
    level=logging.NOTSET,
)


dp = Dispatcher()
dp.include_router(router)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
