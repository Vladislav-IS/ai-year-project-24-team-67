import logging
import json
import httpx
import time
import matplotlib.pyplot as plt
import io
from settings import settings
from states import ProjectStates
from typing import Union
from aiogram.client.default import DefaultBotProperties
from aiogram import F, Router, Bot
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, InlineKeyboardMarkup, \
    InlineKeyboardButton, CallbackQuery, BufferedInputFile


bot = Bot(token=settings.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
router = Router()

train_buttons = [
    [InlineKeyboardButton(text='DL-модель', callback_data='dl_train')],
    [InlineKeyboardButton(text='Классическая ML-модель', callback_data='classic_ml_train')]
    ]
train_markup = InlineKeyboardMarkup(inline_keyboard=train_buttons, row_width=1)

types_dict = {
    'LogReg': 'Логистическая регрессия',
    'RandomForest': 'Случайный лес',
    'GradientBoosting': 'Градиентный бустинг',
    'NeuralNetwork': 'Нейронная сеть',
    'SVM': 'Машина опорных векторов'
}


@router.message(CommandStart())
async def start_handler(message: Message):
    logging.info(f"Start command from user: {message.from_user.id} ({message.from_user.full_name})")
    await message.answer('Доброго времени суток!')


@router.message(F.text == '/eda')
async def eda_handler(message: Message) -> Union[bytes, str]:
    logging.info(f"EDA command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_eda_pdf') 
        if response.status_code == 200:
            data = response.read()
            pdf_file = BufferedInputFile(file=data, filename="EDA.pdf")
            await message.answer_document(pdf_file, caption="Результаты EDA представлены в файле")
        else:
            await message.answer('Повторите свой запрос позже!')


@router.message(F.text == '/train')
async def train_handler(message: Message, state: FSMContext):
    logging.info(f"Train command from user: {message.from_user.id} ({message.from_user.full_name})")
    await message.answer('Выберите тип:', reply_markup=train_markup)
    await state.set_state(ProjectStates.train_type_select)


@router.callback_query(F.data == 'classic_ml_train', ProjectStates.train_type_select)
async def classic_ml_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Classic ML train command from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await callback.message.edit_reply_markup()
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_model_types')
        if response.status_code == 200:
            models = response.json()['models']
            await state.update_data(model_types=models)
            builder = InlineKeyboardBuilder()
            for model in models.keys():
                builder.button(text=types_dict[model], callback_data=f"model_{model}")
            builder.adjust(1)
            await callback.message.answer("Выберите модель для обучения:", reply_markup=builder.as_markup())
            await state.set_state(ProjectStates.model_select)
        else:
            await callback.message.answer("Повторите свой запрос позже!")
            await state.clear()


@router.callback_query(F.data.startswith('model_'), ProjectStates.model_select)
async def select_model_handler(callback: CallbackQuery, state: FSMContext):
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
async def select_hyperparam_handler(callback: CallbackQuery, state: FSMContext):
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
async def hyperparam_value_handler(message: Message, state: FSMContext):
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
async def time_limit_handler(message: Message, state: FSMContext):
    logging.info(f"Time limit input from user: {message.from_user.id} ({message.from_user.full_name})")
    try:
        time_limit = int(message.text.strip())
    except:
        await message.answer("Введите целое число:")
        return
    await state.update_data(time_limit=time_limit)
    user_data = await state.get_data()
    current_model = user_data['current_model']
    if current_model != 'NeuralNetwork':
        await message.answer("Теперь введите ID модели")
        await state.set_state(ProjectStates.model_id_input)
    else:
        await message.answer("Теперь введите размер батча:")
        await state.set_state(ProjectStates.batch_size_input)


@router.message(ProjectStates.batch_size_input)
async def batch_size_handler(message: Message, state: FSMContext):
    logging.info(f"Batch size input from user: {message.from_user.id} ({message.from_user.full_name})")
    try:
        batch_size = int(message.text.strip())
    except:
        await message.answer("Введите целое число:")
        return
    await state.update_data(batch_size=batch_size)
    builder = InlineKeyboardBuilder()
    user_data = await state.get_data()
    for metric in user_data['dl_info']['metrics']:
        builder.button(text=metric, callback_data=f"metric_{metric}")
    builder.adjust(1)
    await message.answer("Выберите метрику:", reply_markup=builder.as_markup())
    await state.set_state(ProjectStates.metric_select)


@router.callback_query(F.data.startswith('metric_'), ProjectStates.metric_select)
async def dl_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Metric selection from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await state.update_data(current_model='NeuralNetwork')
    await callback.message.edit_reply_markup()
    metric_data = '_'.join(callback.data.split('_')[1:])
    await state.update_data(metric=metric_data)
    await callback.message.answer("Теперь введите относительный размер тестовой выборки в пределах 0.1-0.9:")
    await state.set_state(ProjectStates.test_size_input)


@router.message(ProjectStates.test_size_input)
async def batch_size_handler(message: Message, state: FSMContext):
    logging.info(f"Test size input from user: {message.from_user.id} ({message.from_user.full_name})")
    try:
        test_size = float(message.text.strip())
    except:
        await message.answer("Введите десятичную дробь в пределах 0.1-0.9:")
        return
    if test_size < 0.1 or test_size > 0.9:
        await message.answer("Введите десятичную дробь в пределах 0.1-0.9:")
        return
    await state.update_data(test_size=test_size)
    await message.answer("Теперь введите количество эпох обучения:")
    await state.set_state(ProjectStates.epochs_num_input)


@router.message(ProjectStates.epochs_num_input)
async def epochs_num_handler(message: Message, state: FSMContext):
    logging.info(f"Epochs input from user: {message.from_user.id} ({message.from_user.full_name})")
    try:
        epochs_num = int(message.text.strip())
    except:
        await message.answer("Введите целое число:")
        return
    await state.update_data(epochs_num=epochs_num)
    await message.answer("Теперь введите ID модели:")
    await state.set_state(ProjectStates.model_id_input)


@router.message(ProjectStates.model_id_input)
async def model_id_handler(message: Message, state: FSMContext):
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
    await state.set_state(ProjectStates.load_train_file)


@router.message(ProjectStates.load_train_file, F.document)
async def train_file_handler(message: Message, state: FSMContext):
    logging.info(f"Upload train file from user: {message.from_user.id} ({message.from_user.full_name})")
    
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
        
        if user_data['current_model'] != 'NeuralNetwork':
            json_data = {
                'type': user_data['current_model'],
                'id': user_data['model_id'],
                'hyperparameters' : user_data.get('hyperparams', {})
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
                            await state.clear()
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
                    await state.clear()
        else:
            json_data = {
                'type': user_data['current_model'],
                'id': user_data['model_id'],
                'hyperparameters' : user_data.get('hyperparams', {})
                }
            json_data['hyperparameters']['time_limit'] = user_data['time_limit']
            json_data['hyperparameters']['batch_size'] = user_data['batch_size']
            json_data['hyperparameters']['test_size'] = user_data['test_size']
            json_data['hyperparameters']['metric'] = user_data['metric']
            json_data['hyperparameters']['epochs_num'] = user_data['epochs_num']
            files = [
                ("model_str", (None, json.dumps(json_data), "application/json")),
                ("file", ('train.csv', file_content, "text/csv"))
                ]
            async with httpx.AsyncClient() as client:
                response = await client.post(settings.FASTAPI_URL + settings.ROUTE + 'train_dl', files=files)        
                if response.status_code == 200:
                    dl_status = response.json()
                    last_update = 0
                    sent = None
                    while dl_status['status'] == 'training started':
                        current_time = time.time()
                        if current_time - last_update > settings.TIME_INTERVAL:
                            async with httpx.AsyncClient() as client:
                                response = await client.get(settings.FASTAPI_URL + 
                                                            settings.ROUTE + 
                                                            "get_dl_status")
                                if response.status_code != 200:
                                    await message.answer("Ошибка обучения модели. Попробуйте позже!")
                                    state.clear()
                                    return
                                else:
                                    if dl_status != response.json():
                                        dl_status = response.json()
                                        epoch = len(dl_status.get('train_loss', []))
                                        if epoch:
                                            x = list(range(1, epoch + 1))
                                            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                                            ax[0].plot(x, 
                                                       dl_status['train_loss'],
                                                       label='Трен.')
                                            ax[0].plot(x, 
                                                       dl_status['val_loss'],
                                                       label='Вал.')
                                            ax[0].set_xlabel('Эпоха')
                                            ax[0].set_ylabel('Функция потерь')
                                            ax[1].plot(x, 
                                                       dl_status['train_metric'],
                                                       label='Трен.')
                                            ax[1].plot(x, 
                                                       dl_status['val_metric'],
                                                       label='Вал.')
                                            ax[1].set_xlabel('Эпоха')
                                            ax[1].set_ylabel('Метрика')
                                            plt.tight_layout() 
                                            buf = io.BytesIO()
                                            plt.savefig(buf, format='png', dpi=80)
                                            buf.seek(0)
                                            if sent is not None:
                                                await bot.delete_message(sent.chat.id, sent.message_id)
                                            photo = BufferedInputFile(file=buf.read(), filename='photo.png')
                                            sent = await message.answer_photo(photo)
                                            plt.close(fig)
                                            buf.close()
                            last_update = current_time
                    if dl_status['status'] == 'load':
                        await message.answer(
                            f"Модель <b>{dl_status['id']}</b> обучена и установлена для инференса!",
                            parse_mode="HTML")
                        await state.clear()
                    elif dl_status['status'] == 'not trained':
                        await message.answer(
                            f"Обучение модели <b>{dl_status['id']}</b> прервано. Увеличьте максимальное время!",
                            parse_mode="HTML")
                        await state.set_state(ProjectStates.time_limit_input)
                    elif dl_status['status'] == 'error':
                        await message.answer(
                            f"Указаны неверные параметры для модели модели <b>{dl_status['id']}</b>!",
                            parse_mode="HTML")
                        await state.clear()
                else:
                    print(response.json()['details'])
                    await message.answer("Ошибка обучения модели. Попробуйте позже!")
                    await state.clear()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        await message.answer("Ошибка запуска обучения. Попробуйте позже!")
        await state.clear()
   

@router.callback_query(F.data == 'dl_train', ProjectStates.train_type_select)
async def dl_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"DL train select command from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await state.update_data(current_model='NeuralNetwork')
    await callback.message.edit_reply_markup()
    await callback.message.answer("Через Telegram-бот можно обучать только нейросети базовой архитектуры!\n\n"
                                  "<b>Базовая архитектура:</b>\n"
                                  "Linear(30, 128)\n"
                                  "ReLU()\n"
                                  "Dropout(p=0.2)\n"
                                  "Linear(128, 128)\n"
                                  "Linear(128, 1)\n"
                                  "Sigmoid()",
                                  parse_mode='HTML')
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_dl_info')
        if response.status_code == 200:
            dl_info = response.json()
            await state.update_data(dl_info=dl_info)
            builder = InlineKeyboardBuilder()
            for device in dl_info['devices']:
                builder.button(text=device, callback_data=f"device_{device}")
            builder.adjust(1)
            await callback.message.answer("Выберите устройство для обучения:", reply_markup=builder.as_markup())
            await state.set_state(ProjectStates.device_select)
        else:
            await callback.message.answer("Повторите свой запрос позже!")
            await state.clear()


@router.callback_query(F.data.startswith('device_'), ProjectStates.device_select)
async def device_select_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Device select command from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await callback.message.edit_reply_markup()
    user_data = await state.get_data()
    losses = user_data['dl_info']['losses']
    device_data = '_'.join(callback.data.split('_')[1:])
    await state.update_data(hyperparams={
        'device' : device_data,
        'model_params' : 'Baseline',
        })
    builder = InlineKeyboardBuilder()
    for loss in losses:
        builder.button(text=loss, callback_data=f"loss_{loss}")
    builder.adjust(1)
    await callback.message.answer("Выберите функцию потерь:", reply_markup=builder.as_markup())
    await state.set_state(ProjectStates.loss_select)


@router.callback_query(F.data.startswith('loss_'), ProjectStates.loss_select)
async def loss_select_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Loss select command from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await callback.message.edit_reply_markup()
    user_data = await state.get_data()
    loss_data = '_'.join(callback.data.split('_')[1:])
    user_data['hyperparams']['loss_params'] = loss_data
    await state.update_data(hyperparams=user_data['hyperparams'])
    optims = user_data['dl_info']['optimizers']
    builder = InlineKeyboardBuilder()
    for optim in optims.keys():
        builder.button(text=optim, callback_data=f"optim_{optim}")
    builder.adjust(1)
    await callback.message.answer("Выберите оптимизатор:", reply_markup=builder.as_markup())
    await state.set_state(ProjectStates.optim_select)


@router.callback_query(F.data.startswith('optim_'), ProjectStates.optim_select)
async def optimizer_select_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Optimizer selection from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    await callback.message.edit_reply_markup()
    optim_data = '_'.join(callback.data.split("_")[1:])
    user_data = await state.get_data()
    user_data['hyperparams']['optimizer_params'] = {}
    user_data['hyperparams']['optimizer_params']['optimizer_type'] = optim_data
    await state.update_data(hyperparams=user_data['hyperparams'])
    optim_params = user_data['dl_info']['optimizers'][optim_data]
    builder = InlineKeyboardBuilder()
    for param in optim_params.keys():
        builder.button(text=param, callback_data=f'optparam_{param}')
    builder.button(text="Завершить ввод", callback_data='optparam_time')
    builder.adjust(1)
    await state.update_data(keyboard_builder=builder)
    await callback.message.answer(
            'Выберите параметр оптимизатора для настройки или нажмите "Завершить ввод"'
            '(для пропущенных параметров будет установлено значение по умолчанию)',
            reply_markup=builder.as_markup(resize_keyboard=True)
        )
    await state.set_state(ProjectStates.optim_param_select)


@router.callback_query(F.data.startswith('optparam_'), ProjectStates.optim_param_select)
async def optparam_select_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Optimizer parameters selection from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    user_data = await state.get_data()
    optim_data = user_data['hyperparams']['optimizer_params']['optimizer_type']
    param_data = '_'.join(callback.data.split('_')[1:])    
    await callback.message.edit_reply_markup()
    if param_data == 'time':
        await callback.message.answer("Введите максимальное время одной эпохи обучения модели (в секундах):")
        await state.set_state(ProjectStates.time_limit_input)
        return
    
    param_type = user_data["dl_info"]["optimizers"][optim_data][param_data]
    if 'literal' in param_type:
        param_type = '/'.join(param_type.split('/')[1:])
    if param_type == 'bool':
        param_type = 'true/1/yes/да'
    await callback.message.answer(
        f"Введите значение для параметра '<b>{param_data}</b>' ({param_type}):",
        parse_mode="HTML"
    )
    await state.update_data(current_param=param_data)
    await state.set_state(ProjectStates.optim_param_value)


@router.message(ProjectStates.optim_param_value)
async def optparam_value_handler(message: Message, state: FSMContext):
    logging.info(f"Optimizer parameters input from user: {message.from_user.id} ({message.from_user.full_name})")
    user_data = await state.get_data()
    hyperparams = user_data['hyperparams']
    optim_data = hyperparams['optimizer_params']['optimizer_type']
    param_data = user_data["current_param"]
    param_type = user_data["dl_info"]['optimizers'][optim_data][param_data]
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
        
        hyperparams['optimizer_params'][param_data] = value
        await state.update_data(hyperparams=hyperparams)
        
        params_list = "\n".join([f"• <b>{k}</b>: {v}" for k, v 
                                 in hyperparams['optimizer_params'].items() 
                                 if k != 'optimizer_type'])
        await message.answer(
            f"Параметр '<b>{param_data}</b>' установлен: {value}\n\n"
            f"Текущие параметры:\n{params_list}\n\n"
            'Выберите гиперпараметр для настройки или нажмите "Завершить ввод"'
            "(для пропущенных параметров будет установлено значение по умолчанию)",
            parse_mode="HTML",
            reply_markup=user_data['keyboard_builder'].as_markup(resize_keyboard=True)
        )
        await state.set_state(ProjectStates.optim_param_select)
    except Exception as e:
        logging.warning(f"Invalid param value: {value} for {param_type}")
        await message.answer(
            f"Некорректное значение для типа <b>{param_type}</b>. Попробуйте еще раз:",
            parse_mode="HTML"
        )


@router.message(F.text == '/models_list')
async def models_list_handler(message: Message, state: FSMContext):
    logging.info(f"Models list command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'models_list')
        if response.status_code == 200:
            text = ''
            models_list = response.json()['models']
            response = await client.get(settings.FASTAPI_URL + settings.ROUTE + 'get_current_model')
            cur_baseline = response.json()
            for model in models_list:
                text += f'ID: {model['id']}, тип: {types_dict[model['type']]}\n'
                if model['id'] == cur_baseline['baseline']:
                    text += '<b>Это модель-бейзлайн.</b>\n'
                if model['id'] == cur_baseline['current']:
                    text += '<b>Это текущая модель.</b>\n'
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
                    text = f'ID: {model['id']}, тип: {types_dict[model['type']]}\n'
                    builder.button(text=text, callback_data=f"remove_{model['id']}")
            builder.adjust(1)
            await message.answer("Выберите модель, которую необходимо удалить:", 
                                 reply_markup=builder.as_markup())
            await state.set_state(ProjectStates.model_remove)


@router.callback_query(F.data.startswith('remove_'), ProjectStates.model_remove)
async def remove_callback_handler(callback: CallbackQuery, state: FSMContext):
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
async def remove_all_handler(message: Message):
    logging.info(f"Remove all command from user: {message.from_user.id} ({message.from_user.full_name})")
    async with httpx.AsyncClient() as client:
        response = await client.delete(settings.FASTAPI_URL + settings.ROUTE + 'remove_all')
        if response.status_code == 200:
            await message.answer("Все модели удалены!")
        else:    
            await message.answer("Ошибка, попробуйте позднее!")


@router.message(F.text == '/set_model')
async def set_model_handler(message: Message, state: FSMContext):
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
                    text = f'ID: {model['id']}, тип: {types_dict[model['type']]}\n'
                    builder.button(text=text, callback_data=f"set_{model['id']}")
            builder.adjust(1)
            await message.answer("Выберите модель, которую необходимо установить для инференса:", 
                                 reply_markup=builder.as_markup())
            await state.set_state(ProjectStates.model_load)


@router.callback_query(F.data.startswith('set_'), ProjectStates.model_load)
async def set_callback_handler(callback: CallbackQuery, state: FSMContext):
    logging.info(f"Set select model from user: {callback.message.from_user.id} ({callback.message.from_user.full_name})")
    model_id = '_'.join(callback.data.split('_')[1:])
    async with httpx.AsyncClient() as client:
        response = await client.post(settings.FASTAPI_URL + settings.ROUTE + f'set_model/{model_id}')
        if response.status_code == 200:
            await callback.message.edit_reply_markup()
            await callback.message.answer(f"Модель {model_id} установлена для инференса")
            await state.clear()
        else:
            await callback.message.answer(f"Ошибка! Попробуйте позже или установите другую модель!")


@router.message(F.text == '/predict')
async def predict_handler(message: Message, state: FSMContext):
    logging.info(f"Predict command from user: {message.from_user.id} ({message.from_user.full_name})")
    await message.answer(
        "Загрузите файл с данными (CSV):\n\n"
        "<i>Первая строка - заголовки столбцов.</i>",
        parse_mode="HTML"
    )
    await state.set_state(ProjectStates.load_predict_file)


@router.message(ProjectStates.load_predict_file, F.document)
async def predict_file_handler(message: Message, state: FSMContext):
    logging.info(f"Upload predict file from user: {message.from_user.id} ({message.from_user.full_name})")
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
        await message.answer("Старт инференса...")
        
        files = [("file", ('predict.csv', file_content, "text/csv"))]
        async with httpx.AsyncClient() as client:
            response = await client.post(settings.FASTAPI_URL + settings.ROUTE + 'predict_with_file', 
                                        files=files)
            if response.status_code == 200:
                result = response.json()
                buf = io.BytesIO()
                buf.write(f"{result['index_name']},predictions\n".encode('utf-8'))
                for v1, v2 in zip(result['index'], result['predictions']):
                    buf.write(f"{v1},{v2}\n".encode('utf-8'))
                buf.seek(0)       
                buf_file = BufferedInputFile(file=buf.read(), filename='result.csv')
                await message.answer_document(buf_file, caption="Результаты представлены в файле")
            else:
                await message.answer('Повторите свой запрос позже!')
                state.clear()
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        await message.answer("Ошибка предсказания. Попробуйте позже!")
        await state.clear()


@router.message(F.text == '/cancel')
async def cancel_handler(message: Message, state: FSMContext):
    logging.info(f"Cancel command from user: {message.from_user.id} ({message.from_user.full_name})")
    await state.clear()
    await message.answer('Все состояния сброшены')


@router.message(F.text == '/help')
async def help_handler(message: Message, state: FSMContext):
    logging.info(f"Cancel command from user: {message.from_user.id} ({message.from_user.full_name})")
    await message.answer(
        f'''
        <b>Список доступных команд</b>:\n
        /start - стартовая команда;\n
        /eda - получить отчет о EDA в виде PDF;\n
        /train - запустить обучение модели;\n
        /predict - получить предсказания текущей модели;\n
        /models_list - получить список обученных моделей;\n
        /remove - удалить модель;\n
        /remove_all - удалить все модели, кроме baseline-модели;\n
        /set_model - установить модель для инференса;\n
        /cancel - отменить предыдущую команду;\n
        /help - получить справку по командам;\n
        ''',
        parse_mode="HTML"
    )