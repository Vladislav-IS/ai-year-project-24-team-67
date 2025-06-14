import datetime
import logging
import re
import time

import client_funcs
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st


def dl_train_clicked():
    '''
    переход на страницу обучения классических ML-моделей
    '''
    st.session_state.train_task = 1


def sklearn_train_clicked():
    '''
    переход на страницу обучения DL-моделей
    '''
    st.session_state.train_task = 2


def back_clicked():
    '''
    возвращение на стартовую страницу
    '''
    st.session_state.train_task = 0
    st.session_state.dl_task = 0
    st.session_state.sklearn_task = 0
    st.session_state.dl_layers_count = 0
    st.session_state.dl_creating_count = 0
    st.session_state.sklearn_creating_count = 0
    st.cache_data.clear()


def back_sklearn_train_clicked():
    '''
    возвращение на страницу обучения
    '''
    st.session_state.sklearn_task = 0


def back_dl_train_clicked():
    '''
    возвращение на страницу обучения
    '''
    st.session_state.dl_task = 0


def create_exit_button():
    '''
    добавление кнопки перезапуска обучения
    '''
    cols = st.columns(3)
    cols[1].button('Начать заново', 
                   use_container_width=True,
                   on_click=back_dl_train_clicked if st.session_state.train_task == 1 else back_sklearn_train_clicked)


@st.cache_data
def get_train_data():
    '''
    получение данных для обучения моделей:
    списка типов моделей и списка столбцов
    '''
    response = client_funcs.get_columns()
    df_cols_data = response.json()
    response = client_funcs.get_classic_ml_info()
    classic_ml_info = response.json()['models']
    response = client_funcs.get_dl_info()
    dl_info = response.json()
    return classic_ml_info, dl_info, df_cols_data


def create_dl_model(index, layer_types):
    '''
    создание виджетов для добавления новой модели
    '''
    placeholder = st.empty()
    st_cols = placeholder.columns(3)
    layer_type = st_cols[0].selectbox('Выберите тип слоя:',
                                      layer_types.keys(),
                                      key=f'layer_wgt_{index}')
    params = layer_types[layer_type]
    st.session_state[f'layer_{index}'] = {'layer_type': layer_type}
    col_idx = 1
    for param, param_type in params.items():
        if param_type == 'int':
            param_val = st_cols[col_idx].number_input(f'Введите параметр {param}:',
                                                      min_value=1,
                                                      max_value=10000,
                                                      value=10,
                                                      step=1,
                                                      key=f'layer_{param}_{index}')
        elif param_type == 'float':
            param_val = st_cols[col_idx].number_input(f'Введите параметр {param}:',
                                                      min_value=0.0,
                                                      max_value=1.0,
                                                      value=0.1,
                                                      step=0.001,
                                                      key=f'layer_{param}_{index}')
        st.session_state[f'layer_{index}'][param] = param_val
        col_idx += 1


def get_mtype(rus_mtype):
    for k, v in st.session_state.wld.items():
        if v == rus_mtype:
            return k
    return ''


def create_sklearn_model(index, model_types):
    '''
    создание виджетов для добавления новой модели
    '''
    st.divider()
    st.write(f'Модель {index + 1}')
    rus_mtype = st.selectbox('Выберите тип модели:',
                             [st.session_state.wld.get(mtype, mtype) for mtype in model_types.keys()],
                             key=f'select_{index}')
    model_id = st.text_input('Введите ID модели',
                             key=f'input_{index}')
    st.session_state[f'mtype_{index}'] = get_mtype(rus_mtype)
    st.session_state[f'model_id_{index}'] = model_id
    params = model_types[get_mtype(rus_mtype)]
    st.session_state[f'params_{index}'] = {}
    st.info('Введите параметры (или оставьте поля пустыми - тогда будут \
            подставлены параметры по умолчанию).')
    for param, param_type in params.items():
        if param_type == 'int':
            param_val = st.number_input(f'Введите параметр {param}:',
                                        min_value=1,
                                        max_value=1000,
                                        value=10,
                                        step=1,
                                        key=f'{param}_{index}')
        elif param_type == 'float':
            param_val = st.number_input(f'Введите параметр {param}:',
                                        min_value=0.0,
                                        max_value=10.0,
                                        value=0.01,
                                        step=0.001,
                                        key=f'{param}_{index}')
        elif 'literal' in param_type:
            param_list = param_type.split('/')[1:]
            param_val = st.selectbox(f'Введите параметр {param}:',
                                     param_list,
                                     key=f'{param}_{index}')
        else:
            param_val = st.checkbox(f'Установить параметр {param}',
                                    value=False,
                                    key=f'{param}_{index}')
        st.session_state[f'params_{index}'][param] = param_val


def add_clicked():
    '''
    добавление новой модели для обучения или
    сравнения качества
    '''
    if st.session_state.train_task == 1:
        st.session_state.dl_creating_count += 1
    else:
        st.session_state.sklearn_creating_count += 1


def delete_clicked():
    '''
    удаление модели на странице обучения или
    сравнения качества
    '''
    if st.session_state.train_task == 1:
        st.session_state.dl_creating_count -= 1
    else:
        st.session_state.sklearn_creating_count -= 1


def start_train_clicked():
    '''
    запуск обучения моделей
    '''
    if st.session_state.train_task == 1:
        st.session_state.dl_task = 1
    else:
        st.session_state.sklearn_task = 1


@st.cache_data
def load_data(csv_file):
    '''
    загрузка датасета
    '''
    df = pd.read_csv(csv_file)
    return df


def start_page(placeholder):
    '''
    стартовая страница
    '''
    with placeholder.container():
        st.write('Выберите дальнейшее действие.')
        st_cols = st.columns(2)
        st_cols[0].button('Обучить DL-модель',
                          on_click=dl_train_clicked,
                          use_container_width=True)
        st_cols[1].button('Обучить классическую модель (SVM, бустинг и т.д.)',
                          on_click=sklearn_train_clicked,
                          use_container_width=True)


def update_train_plots(train_status):
    fig = make_subplots(rows=1,
                        cols=2,
                        specs=[[{"type": "xy"}, {"type": "xy"}]],
                        subplot_titles=("Функция потерь", "Метрика"))
    epochs = [i + 1 for i in range(len(train_status['train_loss']))]
    loss = st.session_state.loss_params
    metric = st.session_state.metric
    fig.add_trace(go.Scatter(y=train_status['train_loss'], x=epochs, name=f'{loss} (трен.)',
                  line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(y=train_status['val_loss'], x=epochs, name=f'{loss} (вал.)',
                  line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=train_status['train_metric'], x=epochs, name=f'{metric} (трен.)',
                  line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(y=train_status['val_metric'], x=epochs, name=f'{metric} (вал.)',
                  line=dict(color='blue')), row=1, col=2)

    fig['layout']['xaxis']['title'] = 'Эпоха'
    fig['layout']['xaxis2']['title'] = 'Эпоха'

    return fig


def train_dl_res_page():
    '''
    страница результатов обучения
    '''
    # with placeholder.container():
    # st_cols = st.columns(3)
    # st_cols[1].button("Назад", on_click=back_clicked,
    #                  use_container_width=True)
    request = {'id': st.session_state.model_id, 'type': 'NeuralNetwork'}
    hyperparams = {'time_limit': st.session_state['time_limit']}
    hyperparams['device'] = st.session_state['device']
    hyperparams['metric'] = st.session_state['metric']
    hyperparams['epochs_num'] = st.session_state[f'epochs_num']
    hyperparams['test_size'] = st.session_state[f'test_size']
    hyperparams['batch_size'] = st.session_state[f'batch_size']
    hyperparams['loss_params'] = st.session_state[f'loss_params']
    hyperparams['optimizer_params'] = st.session_state[f'optimizer_params']
    if st.session_state.architecture == 'Baseline':
        hyperparams['model_params'] = st.session_state.architecture
    else:
        hyperparams['model_params'] = {
            'layers_count': st.session_state.dl_creating_count}
        for index in range(st.session_state.dl_creating_count):
            hyperparams['model_params'][f'layer_{index}'] = st.session_state[f'layer_{index}']
    request['hyperparameters'] = hyperparams
    response = client_funcs.train_dl_model(request, st.session_state.train_csv)
    if response.status_code == 200:
        train_status = response.json()

        placeholder = st.empty()
        while train_status['status'] == 'training started':
            current_time = time.time()
            if current_time - st.session_state.last_update > st.session_state.time_interval:
                response = client_funcs.get_dl_status()
                if response.status_code != 200:
                    st.error(f'Ошибка, сообщение от сервера: {response.content}')
                else:
                    if train_status != response.json():
                        train_status = response.json()
                        placeholder.plotly_chart(update_train_plots(train_status),
                                                 use_container_width=True,
                                                 theme="streamlit")
                st.session_state.last_update = current_time
        if train_status['status'] == 'trained':
            st.info(f'Модель {train_status['id']} обучена')
        elif train_status['status'] == 'not trained':
            st.error(f'Обучение модели {train_status['id']} прервано')
        elif train_status['status'] == 'load':
            st.info(f'Модель {train_status['id']} установлена для инференса')
        else:
            st.error(f'Указаны неверные параметры \
                     для модели {train_status['id']}')
        create_exit_button()
    else:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')
        create_exit_button()


def train_sklearn_res_page():
    requests = []
    if st.session_state.train_task == 2:
        for index in range(st.session_state.sklearn_creating_count):
            request = {}
            request['type'] = st.session_state[f'mtype_{index}']
            request['id'] = st.session_state[f'model_id_{index}']
            request['hyperparameters'] = {
                'time_limit': st.session_state['time_limit']}
            for param, value in st.session_state[f'params_{index}'].items():
                request['hyperparameters'][param] = value
            requests.append(request)
        responses = client_funcs.train_sklearn_models(
            requests, st.session_state.train_csv)
    if responses.status_code == 200:
        for response in responses.json():
            model_status = response['status']
            model_id = response['id']
            if model_status == 'trained':
                st.info(f'Модель {model_id} обучена')
            elif model_status == 'not trained':
                st.error(f'Обучение модели {model_id} прервано')
            elif model_status == 'load':
                st.info(f'Модель {model_id} установлена для инференса')
            else:
                st.error(
                    f'Указаны неверные параметры \
                        для модели {model_id}')
            create_exit_button()
    else:
        st.error(f'Ошибка, сообщение от сервера: {responses.content}')
        create_exit_button()


def dl_train_page(placeholder, dl_info, df_cols_data):
    '''
    страница обучения DL-моделей
    '''
    placeholder.empty()
    types_list = {
        'Целое число': int,
        'Дробь': float,
        'Строка': str
    }
    with placeholder.container():
        st_cols = st.columns(3)
        st_cols[1].button("Назад", on_click=back_clicked,
                          use_container_width=True)
        train_csv = st.file_uploader(
            'Загрузите тренировочный датасет', type=['csv'])
        if train_csv is not None:
            train_df = load_data(train_csv)
            if not client_funcs.check_dataset(train_df, df_cols_data):
                st.error('Ошибка данных в тренировочном датасете!')
            else:
                model_id = st.text_input('Введите ID модели:', key=f'dl_input')
                st.session_state['model_id'] = model_id
                st.session_state.train_csv = train_csv
                device = st.selectbox('Выберите устройство для обучения:',
                                      dl_info['devices'],
                                      key='devices_wgt')
                st.session_state['device'] = device
                st.warning('При проектировании сети вручную не забудьте правильно учесть \
                           размерность входных данных и формат выходных данных (в зависимости от лосса).')
                arch = st.selectbox('Выберите тип модели (Baseline - использовать стандартную \
                                    архитектуру, Manual - реализовать вручную):',
                                    dl_info['architectures'],
                                    key='architectures_wgt')
                st.session_state['architecture'] = arch
                metric = st.selectbox('Выберите метрику:',
                                      dl_info['metrics'],
                                      key='metric_wgt')
                st.session_state['metric'] = metric
                loss = st.selectbox('Выберите функцию потерь:',
                                    dl_info['losses'],
                                    key='loss_params_wgt')
                st.session_state['loss_params'] = loss
                batch_size = st.number_input('Введите размер батча:',
                                             min_value=1,
                                             max_value=64,
                                             value=32,
                                             step=1,
                                             key='batch_size_wgt')
                st.session_state['batch_size'] = batch_size
                test_size = st.number_input('Введите относительную величину тестовой выборки:',
                                            min_value=0.1,
                                            max_value=0.9,
                                            value=0.1,
                                            step=0.01,
                                            key='test_size_wgt')
                st.session_state['test_size'] = test_size
                epochs_num = st.number_input('Введите количество эпох',
                                             min_value=1,
                                             max_value=1000,
                                             value=10,
                                             step=1,
                                             key='epochs_num_wgt')
                st.session_state['epochs_num'] = epochs_num
                time_limit = st.number_input('Введите максимальное время обучения на одной эпохе:',
                                             min_value=1,
                                             max_value=1000,
                                             value=60,
                                             step=1,
                                             key='time_limit_wgt')
                st.session_state['time_limit'] = time_limit
                optimizer_type = st.selectbox('Выберите оптимизатор:',
                                              dl_info['optimizers'].keys(),
                                              key='optimizer_params_wgt')
                st.session_state['optimizer_params'] = {
                    'optimizer_type': optimizer_type}
                st.divider()
                st.write(f'Параметры оптимизатора')
                for param, param_type in dl_info['optimizers'][optimizer_type].items():
                    if param_type == 'int':
                        param_val = st.number_input(f'Введите параметр {param}:',
                                                    min_value=1,
                                                    max_value=1000,
                                                    value=10,
                                                    step=1,
                                                    key=f'opt_{param}:')
                    elif param_type == 'float':
                        param_val = st.number_input(f'Введите параметр {param}:',
                                                    min_value=0.0,
                                                    max_value=10.0,
                                                    value=0.01,
                                                    step=0.001,
                                                    key=f'opt_{param}')
                    elif 'literal' in param_type:
                        param_list = param_type.split('/')[1:]
                        param_val = st.selectbox(f'Введите параметр {param}:',
                                                 param_list,
                                                 key=f'opt_{param}')
                    else:
                        param_val = st.checkbox(f'Установить параметр {param}',
                                                value=False,
                                                key=f'opt_{param}')
                    st.session_state[f'optimizer_params'][param] = param_val
                if arch == 'Baseline':
                    st_cols = st.columns(3)
                    st_cols[1].button('Начать обучение',
                                      on_click=start_train_clicked,
                                      disabled=st.session_state.dl_task==1,
                                      use_container_width=True)
                else:
                    st.divider()
                    st.write(f'Архитектура сети')
                    with st.container():
                        for i in range(st.session_state.dl_creating_count):
                            create_dl_model(i, dl_info['layers'])
                    disabled = st.session_state.dl_creating_count == 0
                    st_cols = st.columns(3)
                    st_cols[0].button('Добавить слой',
                                      on_click=add_clicked,
                                      use_container_width=True)
                    st_cols[1].button('Удалить слой',
                                      on_click=delete_clicked,
                                      disabled=disabled,
                                      use_container_width=True)
                    st_cols[2].button('Начать обучение',
                                      on_click=start_train_clicked,
                                      disabled=disabled or st.session_state.dl_task==1,
                                      use_container_width=True)
        if st.session_state.dl_task == 1:
            # placeholder.empty()
            if st.session_state.model_id == '':
                time_id = re.sub(r"\D", "", str(datetime.datetime.now()))
                st.session_state.model_id = f'NeuralNetwork_{time_id}'
            train_dl_res_page()


def sklearn_train_page(placeholder, model_types, df_cols_data):
    '''
    страница обучения классических ML-моделей
    '''
    placeholder.empty()
    with placeholder.container():
        st_cols = st.columns(3)
        st_cols[1].button("Назад", on_click=back_clicked,
                          use_container_width=True)
        train_csv = st.file_uploader(
            'Загрузите тренировочный датасет', type=['csv'])
        if train_csv is not None:
            train_df = load_data(train_csv)
            if not client_funcs.check_dataset(train_df, df_cols_data):
                st.error('Ошибка данных в тренировочном датасете!')
            else:
                st.session_state.train_csv = train_csv
                time_limit = st.number_input('Введите максимальное время обучения одной модели (в секундах):',
                                             min_value=1,
                                             max_value=1000,
                                             value=60,
                                             step=1,
                                             key='time_limit_wgt')
                st.session_state['time_limit'] = time_limit
                with st.container():
                    for i in range(st.session_state.sklearn_creating_count):
                        create_sklearn_model(i, model_types)
                disabled = st.session_state.sklearn_creating_count == 0
                st_cols = st.columns(3)
                st_cols[0].button('Добавить модель',
                                  on_click=add_clicked,
                                  use_container_width=True)
                st_cols[1].button('Удалить модель',
                                  on_click=delete_clicked,
                                  disabled=disabled,
                                  use_container_width=True)
                st_cols[2].button('Начать обучение',
                                  on_click=start_train_clicked,
                                  disabled=disabled or st.session_state.sklearn_task==1,
                                  use_container_width=True)
        if st.session_state.sklearn_task == 1:
            # placeholder.empty()
            for i in range(st.session_state.sklearn_creating_count):
                if st.session_state[f'model_id_{i}'] == '':
                    time_id = re.sub(r"\D", "", str(datetime.datetime.now()))
                    st.session_state[f'model_id_{i}'] = \
                        f'{st.session_state[f'mtype_{i}']}_{i + 1}_{time_id}'
            train_sklearn_res_page()


@st.cache_data
def to_csv(df):
    '''
    чтение датасета из csv-файла
    '''
    return df.to_csv().encode('utf-8')


@st.cache_resource
def make_predictions(csv):
    response = client_funcs.predict(csv)
    if response.status_code == 200:
        preds = response.json()
        df = pd.DataFrame({
            'predictions': preds['predictions']
        }, index=preds['index'])
        df.index.name = preds['index_name']
        return True, df
    else:
        return False, response.content


logging.info('Classic ML opened')

classic_ml_info, dl_info, df_cols_data = get_train_data()

# переменная с текущим типом страниц
if 'train_task' not in st.session_state:
    st.session_state.train_task = 0

# переменная состояния страницы обучения DL-моделей
if 'dl_task' not in st.session_state:
    st.session_state.dl_task = 0

# переменная состояния страницы обучения классических ML-моделей
if 'sklearn_task' not in st.session_state:
    st.session_state.sklearn_task = 0

# количество DL-моделей, созданных для обучения
if 'dl_creating_count' not in st.session_state:
    st.session_state.dl_creating_count = 0

# количество классических ML-моделей, созданных для обучения
if 'sklearn_creating_count' not in st.session_state:
    st.session_state.sklearn_creating_count = 0

# таймер обучения DL-модели
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0

# интервал запросов статуса обучения (в секундах)
if 'time_interval' not in st.session_state:
    st.session_state.time_interval = 1

# словарь переводов
# словарь переводов
if 'wld' not in st.session_state:
    st.session_state.wld = {
        'NeuralNetwork': 'Нейронная сеть',
        'LogReg': 'Логистическая регрессия',
        'SVM': 'Машина опорных векторов',
        'RandomForest': 'Случайный лес',
        'GradientBoosting': 'Градиентный бустинг'
    }

placeholder = st.empty()

if st.session_state.train_task == 0:
    logging.info('Start page opened')
    start_page(placeholder)
elif st.session_state.train_task == 1:
    logging.info('DL train page opened')
    dl_train_page(placeholder, dl_info, df_cols_data)
elif st.session_state.train_task == 2:
    logging.info('Classic ML page opened')
    sklearn_train_page(placeholder, classic_ml_info, df_cols_data)
