from aiogram import F, Router, types
from aiogram.filters.command import Command

router = Router()


@router.message(F.text, Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Бот готов к использованию!")


@router.message(F.text, Command("help"))
async def cmd_help(message: types.Message):
    await message.reply(
        '''
        Доступные команды:
        /start - начало работы
        /get_eda - просмотр EDA
        /train_model - обучить модель
        /models_list - получить список обученных моделей
        /set_model - установить модель для инференса
        /predict - получить предсказание модели
        /help - справка
        '''
    )


@router.message(F.text, Command("get_eda"))
async def cmd_get_eda(message: types.Message):
    pass
