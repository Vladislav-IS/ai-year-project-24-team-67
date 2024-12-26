import asyncio
import logging
from aiogram import Bot, Dispatcher
from config import TOKEN
from handlers import router


logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher()
dp.include_router(router)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())