import logging
import json
import httpx
from settings import settings
from states import ProjectStates
from typing import Union
from aiogram.client.default import DefaultBotProperties
from aiogram import F, Router, Bot
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state
from aiogram.types import Message, InlineKeyboardMarkup, \
    InlineKeyboardButton, CallbackQuery, BufferedInputFile


bot = Bot(token=settings.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
router = Router()

train_buttons = [
    [InlineKeyboardButton(text='DL-модель', callback_data='dl_train')],
    [InlineKeyboardButton(text='Классическая ML-модель', callback_data='classic_ml_train')]
    ]

train_markup = InlineKeyboardMarkup(inline_keyboard=train_buttons, row_width=1)


@router.message(CommandStart())
async def cmd_start(message: Message):
    logging.info(f"Start command from user: {message.from_user.id} ({message.from_user.full_name})")
    await message.answer('Доброго времени суток!')


@router.message(F.text == '/eda')
async def cmd_eda(message: Message) -> Union[bytes, str]:
    logging.info(f"EDA command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_eda_pdf') 
        if response.status_code == 200:
            data = response.read()
            pdf_file = BufferedInputFile(file=data, filename="EDA.pdf")
            await message.answer_document(pdf_file)
        else:
            await message.answer('Повторите свой запрос позже!')


@router.message(F.text == '/train')
async def cmd_train(message: Message, state: FSMContext):
    logging.info(f"Train command from user: {message.from_user.id} ({message.from_user.full_name})")
    await message.answer('Выберите тип:', reply_markup=train_markup)
    await state.set_state(ProjectStates.train_type_select)


@router.callback_query(F.data == 'classic_ml_train', ProjectStates.train_type_select)
async def cmd_classic_ml_train(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Classic ML train command from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await callback.message.edit_reply_markup()
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_model_types')
        if response.status_code == 200:
            models = response.json()['models']
            await state.update_data(model_types=models)
            builder = InlineKeyboardBuilder()
            for model in models.keys():
                builder.button(text=model, callback_data=f"model_{model}")
            builder.adjust(1)
            await callback.message.answer("Выберите модель для обучения:", reply_markup=builder.as_markup())
            await state.set_state(ProjectStates.model_select)
        else:
            await callback.message.answer("Повторите свой запрос позже!")
            await state.clear()


@router.callback_query(F.data.startswith('model_'), ProjectStates.model_select)
async def select_model(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Model selection from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await callback.message.edit_reply_markup()
    model = callback.data.split("_")[1]
    await state.update_data(current_model=model)
    user_data = await state.get_data()
    hyperparams = user_data['model_types'][model]
    builder = InlineKeyboardBuilder()
    for param in hyperparams.keys():
        builder.button(text=param, callback_data=f'param_{param}')
    builder.button(text="Завершить ввод", callback_data='param_time')
    builder.adjust(2)
    await state.update_data(keyboard_builder=builder)
    await callback.message.answer(
            'Выберите гиперпараметр для настройки или нажмите "Завершить ввод"'
            '(для пропущенных параметров будет установлено значение по умолчанию)',
            reply_markup=builder.as_markup(resize_keyboard=True)
        )
    await state.set_state(ProjectStates.hyperparam_select)
    await state.update_data(
        hyperparams={},
        param_types=hyperparams
    )


@router.callback_query(F.data.startswith('param'), ProjectStates.hyperparam_select)
async def select_hyperparam(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Model parameters selection from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    user_data = await state.get_data()
    param_data = '_'.join(callback.data.split('_')[1:])    
    await callback.message.edit_reply_markup()
    if param_data == 'time':
        await callback.message.answer("Введите максимальное время обучения модели (в секундах):")
        await state.set_state(ProjectStates.time_limit_input)
        return
    
    param_type = user_data["param_types"][param_data]
    if 'literal' in param_type:
        param_type = '/'.join(param_type.split('/')[1:])
    if param_type == 'bool':
        param_type = 'true/1/yes/да'
    await callback.message.answer(
        f"Введите значение для параметра '<b>{param_data}</b>' ({param_type}):",
        parse_mode="HTML"
    )
    await state.update_data(current_param=param_data)
    await state.set_state(ProjectStates.hyperparam_value)


@router.message(ProjectStates.hyperparam_value)
async def set_hyperparam_value(message: Message, state: FSMContext):
    logging.info(f"Model parameters input from user: {message.from_user.id} ({message.from_user.full_name})")
    user_data = await state.get_data()
    param_data = user_data["current_param"]
    param_type = user_data["param_types"][param_data]
    value = message.text
    
    try:
        if param_type == "int":
            value = int(value)
        elif param_type == "float":
            value = float(value)
        elif 'literal' in param_type:
            if value not in param_type.split('/'):
                raise ValueError("Unknown literal value")
        elif param_type == "bool":
            value = value.lower() in ["true", "1", "yes", "да"]
        else:
            raise ValueError(f"Unknown type: {param_type}")
        
        hyperparams = user_data["hyperparams"]
        hyperparams[param_data] = value
        await state.update_data(hyperparams=hyperparams)
        
        params_list = "\n".join([f"• <b>{k}</b>: {v}" for k, v in hyperparams.items()])
        await message.answer(
            f"Параметр '<b>{param_data}</b>' установлен: {value}\n\n"
            f"Текущие параметры:\n{params_list}\n\n"
            'Выберите гиперпараметр для настройки или нажмите "Завершить ввод"'
            "(для пропущенных параметров будет установлено значение по умолчанию)",
            parse_mode="HTML",
            reply_markup=user_data['keyboard_builder'].as_markup(resize_keyboard=True)
        )
        await state.set_state(ProjectStates.hyperparam_select)
    except:
        logging.warning(f"Invalid param value: {value} for {param_type}")
        await message.answer(
            f"Некорректное значение для типа <b>{param_type}</b>. Попробуйте еще раз:",
            parse_mode="HTML"
        )


@router.message(ProjectStates.time_limit_input)
async def input_time_limit(message: Message, state: FSMContext):
    logging.info(f"Time limit input from user: {message.from_user.id} ({message.from_user.full_name})")
    try:
        time_limit = int(message.text.strip())
    except:
        await message.answer("Введите целое число:")
        return
    await state.update_data(time_limit=time_limit)
    await message.answer("Теперь введите ID модели")
    await state.set_state(ProjectStates.model_id_input)


@router.message(ProjectStates.model_id_input)
async def input_model_id(message: Message, state: FSMContext):
    logging.info(f"Model ID input from user: {message.from_user.id} ({message.from_user.full_name})")
    model_id = message.text.strip()
    if not model_id:
        await message.answer("ID модели не может быть пустым. Введите еще раз:")
        return
    await state.update_data(model_id=model_id)
    await message.answer(
        "Теперь загрузите файл с обучающими данными (CSV):\n\n"
        "<i>Файл должен содержать данные для обучения модели. "
        "Первая строка - заголовки столбцов.</i>",
        parse_mode="HTML"
    )
    await state.set_state(ProjectStates.load_file)


@router.message(ProjectStates.load_file, F.document)
async def handle_data_file(message: Message, state: FSMContext):
    logging.info(f"Upload file from user: {message.from_user.id} ({message.from_user.full_name})")
    
    user_data = await state.get_data()
    
    try:
        mime_type = message.document.mime_type
        if mime_type not in ["text/csv"]:
            await message.answer("Неподдерживаемый формат файла. Пожалуйста, загрузите CSV-файл.")
            return
        
        file = await bot.get_file(message.document.file_id)
        file_content = await bot.download_file(file.file_path)
        
        await bot.send_chat_action(
            chat_id=message.chat.id,
            action="typing"
        )
        await message.answer("Начинаю обучение модели...")
        
        json_data = {
            'type': user_data['current_model'],
            'id': user_data['model_id'],
            'hyperparameters' : user_data.get('hyperparameters', {})
            }
        json_data['hyperparameters']['time_limit'] = user_data['time_limit']
        files = [
            ("models_str", (None, json.dumps([json_data]), "application/json")),
            ("file", ('train.csv', file_content, "text/csv"))
            ]
        async with httpx.AsyncClient() as client:
            response = await client.post(settings.FASTAPI_URL + settings.ROUTE + 'train_classic_ml', files=files)        
            if response.status_code == 200:
                models_info = response.json()
                for info in models_info:
                    if info['status'] == 'load':
                        await message.answer(
                            f"Модель <b>{info['id']}</b> обучена и установлена для инференса!",
                            parse_mode="HTML")
                    elif info['status'] == 'not trained':
                        await message.answer(
                            f"Обучение модели <b>{info['id']}</b> прервано. Увеличьте максимальное время!",
                            parse_mode="HTML")
                        await state.set_state(ProjectStates.time_limit_input)
                    elif info['status'] == 'error':
                        await message.answer(
                            f"Указаны неверные параметры для модели модели <b>{info['id']}</b>!",
                            parse_mode="HTML")
                        await state.clear()
            else:
                await message.answer("Ошибка обучения модели. Попробуйте позже!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        await message.answer("Ошибка запуска обучения. Попробуйте позже!")
        await state.clear()


@router.message(F.text == '/models_list')
async def remove_handler(message: Message, state: FSMContext):
    logging.info(f"Models list command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'models_list')
        if response.status_code == 200:
            text = ''
            models_list = response.json()['models']
            response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_current_model')
            for model in models_list:
                text += f'ID: {model['id']}, тип: {model['type']}'
                if len(model['hyperparameters']):
                    text = text + "\nПараметры:\n"
                    for param, param_value in model['hyperparameters'].items():
                        text += f'{param}={param_value}\n'
                    text = text[:-1]
                text = text + '\n\n'
            await message.answer("<b>Список моделей</b>:\n\n" + text[:-2], 
                                 parse_mode='HTML')
            await state.set_state(ProjectStates.model_remove)


@router.message(F.text == '/remove')
async def remove_handler(message: Message, state: FSMContext):
    logging.info(f"Remove command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'models_list')
        if response.status_code == 200:
            builder = InlineKeyboardBuilder()
            models_list = response.json()['models']
            response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_current_model')
            baseline_id = '' if response.status_code != 200 else response.json()['baseline']
            for model in models_list:
                if model['id'] != baseline_id:
                    text = f'ID: {model['id']}, тип: {model['type']}\n'
                    #for param, param_value in model['hyperparameters'].items():
                    #    text += f'{param}={param_value},'
                    builder.button(text=text, callback_data=f"remove_{model['id']}")
            builder.adjust(1)
            await message.answer("Выберите модель, которую необходимо удалить:", 
                                 reply_markup=builder.as_markup())
            await state.set_state(ProjectStates.model_remove)


@router.message(F.data.startswith('remove'))
async def remove_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Remove select model from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    model_id = '_'.join(callback.data.split('_')[1:])
    async with httpx.AsyncClient() as client:
        response = await client.delete(settings.FASTAPI_URL + settings.ROUTE + f'remove/{model_id}')
        if response.status_code == 200:
            await callback.message.edit_reply_markup()
            await callback.message.answer(f"Модель {model_id} удалена")
            await state.clear()
        else:
            await callback.message.answer(f"Ошибка! Попробуйте позже или удалите другую модель!")


@router.message(F.text == '/remove_all')
async def rmove_all_handler(message: Message):
    logging.info(f"Remove all command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.delete(settings.FASTAPI_URL + settings.ROUTE + 'remove_all')
        if response.status_code == 200:
            await message.answer("Все модели удалены!")
        else:    
            await message.answer("Ошибка, попробуйте позднее!")


@router.message(F.text == '/set_model')
async def set_handler(message: Message, state: FSMContext):
    logging.info(f"Set command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'models_list')
        if response.status_code == 200:
            builder = InlineKeyboardBuilder()
            models_list = response.json()['models']
            response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_current_model')
            current_id = '' if response.status_code != 200 else response.json()['current']
            for model in models_list:
                if model['id'] != current_id:
                    text = f'ID: {model['id']}, тип: {model['type']}\n'
                    #for param, param_value in model['hyperparameters'].items():
                    #    text += f'{param}={param_value},'
                    builder.button(text=text, callback_data=f"set_{model['id']}")
            builder.adjust(1)
            await message.answer("Выберите модель, которую необходимо установить для инференса:", 
                                 reply_markup=builder.as_markup())
            await state.set_state(ProjectStates.model_load)


@router.message(F.data.startswith('set'))
async def remove_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Set select model from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    model_id = '_'.join(callback.data.split('_')[1:])
    async with httpx.AsyncClient() as client:
        response = await client.delete(settings.FASTAPI_URL + settings.ROUTE + f'set_model/{model_id}')
        if response.status_code == 200:
            await callback.message.edit_reply_markup()
            await callback.message.answer(f"Модель {model_id} установлена для инференса")
            await state.clear()
        else:
            await callback.message.answer(f"Ошибка! Попробуйте позже или установите другую модель!")


@router.message(F.text == '/cancel')
async def cancel_handler(message: Message, state: FSMContext):
    logging.info(f"Cancel command from user: {message.from_user.id} ({message.from_user.full_name})")
    await state.clear()
    await message.answer('Все состояния сброшены')


@router.message(F.text == '/help')
async def cancel_handler(message: Message, state: FSMContext):
    logging.info(f"Cancel command from user: {message.from_user.id} ({message.from_user.full_name})")
    await message.answer(
        f'''
        <b>Список доступных команд</b>:\n
        /start - стартовая команда;\n
        /eda - получить отчет о EDA в виде PDF;\n
        /train - запустить обучение модели;\n
        /models_list - получить список обученных моделей;\n
        /remove - удалить модель;\n
        /remove_all - удалить все модели, кроме baseline-модели;\n
        /set_model - установить модель для инференса;\n
        /cancel - отменить предыдущую команду;\n
        /help - получить справку по командам;\n
        ''',
        parse_mode="HTML"
    )