import logging

import client_funcs
import pandas as pd
import plotly.express as px
import streamlit as st


def train_clicked():
    '''
    переход на страницу обучения моделей
    '''
    st.session_state.models_task = 1


def predict_clicked():
    '''
    переход на страницу предсказаний
    '''
    st.session_state.models_task = 2


def list_clicked():
    '''
    переход на страницу списка моделей
    '''
    st.session_state.models_task = 3


def back_clicked():
    '''
    возвращение на стартову страницу
    '''
    st.session_state.models_task = 0
    st.session_state.train_task = 0
    st.session_state.list_task = 0
    st.cache_data.clear()


@st.cache_data
def get_train_data():
    '''
    получение данных для обучения моделей:
    списка типов моделей и списка столбцов
    '''
    response = client_funcs.get_columns()
    df_cols_data = response.json()
    response = client_funcs.get_model_types()
    model_types = response.json()['models']
    return model_types, df_cols_data


def create_model(index, model_types, types_list):
    '''
    создание виджетов для добавления новой модели
    '''
    st.divider()
    st.write(f'Модель {index + 1}')
    mtype = st.selectbox('Выберите тип модели:',
                         model_types.keys(),
                         key=f'select_{index}')
    model_id = st.text_input('Введите ID модели',
                             key=f'input_{index}')
    st.session_state[f'mtype_{index}'] = mtype
    st.session_state[f'model_id_{index}'] = model_id
    params = model_types[mtype]
    st.session_state[f'params_{index}'] = {}
    st.session_state[f'ptypes_{index}'] = {}
    st.info('Введите параметры (или оставьте поля пустыми - тогда будут \
            подставлены параметры по умолчанию).')
    for param in params:
        st_cols = st.columns(2)
        value = st_cols[0].text_input(f'{param}',
                                      key=f'text_{index}{param}')
        ptype = st_cols[1].selectbox('Тип параметра:',
                                     types_list,
                                     key=f'type_{index}{param}')
        if value is not None:
            st.session_state[f'params_{index}'][param] = value
            st.session_state[f'ptypes_{index}'][param] = ptype


def choose_model(index):
    '''
    создание виджетов для выбора модели из списка
    с целью сравнения качества
    '''
    model_id = st.selectbox('Выберите модель для сравнения',
                            st.session_state.model_ids,
                            key=f"choose_{index}")
    st.session_state[f'choosing_id_{index}'] = model_id


def add_clicked():
    '''
    добавление новой модели для обучения или
    сравнения качества
    '''
    if st.session_state.models_task == 1:
        st.session_state.creating_count += 1
    else:
        st.session_state.choosing_count += 1


def delete_clicked():
    '''
    удаление модели на странице обучения или
    сравнения качества
    '''
    if st.session_state.models_task == 1:
        st.session_state.creating_count -= 1
    else:
        st.session_state.choosing_count -= 1


def start_train_clicked():
    '''
    запуск обучения моделей
    '''
    st.session_state.train_task = 1


def start_choose_clicked():
    '''
    запуск сравнения качества моделей
    '''
    st.session_state.choose_task = 1


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
        st_cols = st.columns(3)
        st_cols[0].button('Обучить модель',
                          on_click=train_clicked,
                          use_container_width=True)
        st_cols[1].button('Сделать предсказание',
                          on_click=predict_clicked,
                          use_container_width=True)
        st_cols[2].button('Список моделей',
                          on_click=list_clicked,
                          use_container_width=True)


def convert(str_param, ptype, types_list):
    '''
    конвертация параметра модели к выбранному типу
    '''
    try:
        return types_list[ptype](str_param.replace(',', '.'))
    except Exception:
        return str_param


def train_res_page(placeholder, types_list):
    '''
    страница результатов обучения
    '''
    with placeholder.container():
        st_cols = st.columns(3)
        st_cols[1].button("Назад", on_click=back_clicked,
                          use_container_width=True)
        requests = []
        for index in range(st.session_state.creating_count):
            request = {}
            request['type'] = st.session_state[f'mtype_{index}']
            request['id'] = st.session_state[f'model_id_{index}']
            request['hyperparameters'] = {}
            for param, value in st.session_state[f'params_{index}'].items():
                ptype = st.session_state[f'ptypes_{index}'][param]
                request['hyperparameters'][param] =\
                    convert(value, ptype, types_list)
            requests.append(request)
        responses = client_funcs.train_models(
            requests, st.session_state.train_csv)
        if responses.status_code == 200:
            for response in responses.json():
                if response['status'] == 'trained':
                    st.info(f'Модель {response['id']} обучена')
                elif response['status'] == 'not trained':
                    st.error(f'Обучение модели {response['id']} прервано')
                else:
                    st.error(
                        f'Указаны неверные параметры \
                            для модели {response['id']}')
        else:
            st.error(f'Ошибка, сообщение от сервера: {responses.content}')
        st.session_state.train_task = 0
        st.session_state.creating_count = 0


def train_page(placeholder, model_types, df_cols_data):
    '''
    страница обучения моделей
    '''
    placeholder.empty()
    types_list = {
        'Целое число': int,
        'Дробь': float,
        'Строка': str
    }
    if st.session_state.train_task == 0:
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
                    for i in range(st.session_state.creating_count):
                        create_model(i, model_types, types_list)
                    disabled = st.session_state.creating_count == 0
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
                                      disabled=disabled,
                                      use_container_width=True)
    elif st.session_state.train_task == 1:
        placeholder.empty()
        empty_ids = []
        for i in range(st.session_state.creating_count):
            if st.session_state[f'model_id_{i}'] == '':
                empty_ids.append(str(i + 1))
        if len(empty_ids):
            empty_ids = ', '.join(empty_ids)
            st_cols = st.columns(3)
            st_cols[1].button("Назад", on_click=back_clicked,
                              use_container_width=True)
            st.error(f'Отсутствует id y моделей №: {empty_ids}')
        else:
            train_res_page(placeholder, types_list)


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


def predict_page(placeholder, df_cols_data):
    '''
    страница предсказаний
    '''
    placeholder.empty()
    st_cols = st.columns(3)
    st_cols[1].button("Назад", on_click=back_clicked, use_container_width=True)
    response = client_funcs.get_current_model()
    if response.status_code == 200:
        if st.session_state.predict_task == 0:
            predict_csv = st.file_uploader(
                'Загрузите датасет для предсказаний', type=['csv'])
            if predict_csv is not None:
                predict_df = load_data(predict_csv)
                if not client_funcs.check_dataset(predict_df,
                                                  df_cols_data,
                                                  'test'):
                    st.error('Ошибка данных в датасете для предсказаний!')
                else:
                    st.session_state.predict_task = 1
                    st.session_state.predict_csv = predict_csv
                    st.rerun()
        elif st.session_state.predict_task == 1:
            res_status, result = make_predictions(st.session_state.predict_csv)
            if res_status:
                st.write('Предсказания модели представлены ниже.')
                st.dataframe(result, use_container_width=True)
                csv = to_csv(result)
                st_cols = st.columns(3)
                st_cols[1].download_button(
                    "Сохранить",
                    data=csv,
                    file_name='predictons.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.error(f'Ошибка, сообщение от сервера: {result}')
    else:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def set_clicked():
    '''
    переход на страницу установки модели для инференса
    '''
    st.session_state.list_task = 1


def unset_clicked():
    '''
    снятие модели с инференса
    '''
    response = client_funcs.unset_model()
    if response.status_code != 200:
        st.error('Текущая модель не найдена')


def remove_model_clicked():
    '''
    переход на страницу удаления модели
    '''
    st.session_state.list_task = 2


def remove_all_clicked():
    '''
    очистка списка моделей
    '''
    response = client_funcs.remove_all()
    if response.status_code != 200:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def set_ok_clicked():
    '''
    подтверждение установки выбранного ID для инференса
    '''
    response = client_funcs.set_model(st.session_state.mid)
    st.session_state.list_task = 0
    if response.status_code != 200:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def remove_ok_clicked():
    '''
    подтверждение удаления модели с выбранным ID
    '''
    response = client_funcs.remove_model(st.session_state.mid)
    st.session_state.list_task = 0
    if response.status_code != 200:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def compare_models_click():
    '''
    переход на страницу сравнения качества моделей
    '''
    st.session_state.list_task = 3


def form_list(models, cur_model):
    '''
    создание виджетов для отображения списка моделей
    '''
    st.session_state.model_ids = []
    for model in models:
        st.divider()
        st.subheader(f'Модель {model['id']}')
        if model['id'] == cur_model:
            st.write('Это текущая модель')
        st.session_state.model_ids.append(model['id'])
        st_cols = st.columns(2)
        st_cols[0].write('Тип:')
        st_cols[1].write(f'{model['type']}')
        for param, value in model['hyperparameters'].items():
            st_cols[0].write(param)
            st_cols[1].write(value)
    st.divider()
    st_cols = st.columns(2)
    st_cols[0].button('Установить текущую модель',
                      on_click=set_clicked,
                      use_container_width=True)
    st_cols[1].button('Снять текущую модель с инференса',
                      on_click=unset_clicked,
                      disabled=cur_model == '',
                      use_container_width=True)
    st_cols[0].button('Удалить модель',
                      on_click=remove_model_clicked,
                      disabled=len(models) == 0,
                      use_container_width=True)
    st_cols[1].button('Удалить все модели',
                      on_click=remove_all_clicked,
                      disabled=len(models) == 0,
                      use_container_width=True)
    st_cols = st.columns([1, 2, 1])
    st_cols[1].button('Сравнить модели',
                      on_click=compare_models_click,
                      use_container_width=True)


@st.cache_data
def draw_hist(results):
    '''
    график качества моделей с разбивкой по ID
    '''
    df = pd.DataFrame()
    for scoring, res in results.items():
        score_df = pd.DataFrame({
            'id': list(res.keys()),
            'value': list(res.values()),
            'scoring': scoring
        })
        df = pd.concat([df, score_df], ignore_index=True)
    fig = px.bar(df, x='id', y='value', color='scoring', barmode='group')
    return fig


def compare_models_page(placeholder, df_cols_data):
    '''
    страница сравнения качества моделей
    '''
    placeholder.empty()
    st_cols = st.columns(3)
    st_cols[1].button("Назад", on_click=back_clicked, use_container_width=True)
    st.write('Здесь вы можете измерить качество одной или нескольких моделей.')
    if st.session_state.choose_task == 0:
        predict_csv = st.file_uploader(
            'Загрузите датасет для сравнения моделей', type=['csv'])
        if predict_csv is not None:
            test_df = load_data(predict_csv)
            if not client_funcs.check_dataset(test_df, df_cols_data):
                st.error('Ошибка данных в тестовом датасете!')
            else:
                st.session_state.predict_csv = predict_csv
                for i in range(st.session_state.choosing_count):
                    choose_model(i)
                add_button_disabled = st.session_state.choosing_count == len(
                    st.session_state.model_ids)
                other_buttons_disabled = st.session_state.choosing_count == 0
                st_cols = st.columns(3)
                st_cols[0].button('Добавить модель',
                                  on_click=add_clicked,
                                  disabled=add_button_disabled,
                                  use_container_width=True)
                st_cols[1].button('Удалить модель',
                                  on_click=delete_clicked,
                                  disabled=other_buttons_disabled,
                                  use_container_width=True)
                st_cols[2].button('Получить результат',
                                  on_click=start_choose_clicked,
                                  disabled=other_buttons_disabled,
                                  use_container_width=True)
    elif st.session_state.choose_task == 1:
        ids = set()
        for i in range(st.session_state.choosing_count):
            ids.add(st.session_state[f'choosing_id_{i}'])
        response = client_funcs.compare_models({'ids': list(ids)},
                                               st.session_state.predict_csv)
        st.plotly_chart(draw_hist(response.json()['results']),
                        use_container_width=True,
                        theme="streamlit")
        st.session_state.choose_task = 0
        st.session_state.choosing_count = 0


def list_page(placeholder, cols):
    '''
    страница списка моделей
    '''
    placeholder.empty()
    if st.session_state.list_task == 0:
        response = client_funcs.get_current_model()
        cur_model = ''
        if response.status_code == 200:
            cur_model = response.json()['message']
        response = client_funcs.get_models_list()
        with placeholder.container():
            st_cols = st.columns(3)
            st_cols[1].button("Назад", on_click=back_clicked,
                              use_container_width=True)
            if response.status_code == 200:
                st.header('Список моделей')
                models = response.json()['models']
                form_list(models, cur_model)
            else:
                st.error('Нет обученных моделей')
    elif st.session_state.list_task in [1, 2]:
        placeholder.empty()
        st_cols = st.columns(3)
        st_cols[1].button("Назад", on_click=back_clicked,
                          use_container_width=True)
        with placeholder.container():
            st.session_state.mid = st.selectbox(
                'Выберите id модели',
                st.session_state.model_ids
            )
            click_fun = set_ok_clicked if st.session_state.list_task == 1\
                else remove_ok_clicked
            st_cols = st.columns(3)
            st_cols[1].button('Принять',
                              on_click=click_fun,
                              use_container_width=True)
    elif st.session_state.list_task == 3:
        compare_models_page(placeholder, cols)


logging.info('Classic ML opened')
st.set_page_config(layout='wide',
                   page_title='Classic ML',
                   page_icon='🤖')
st.title("Classic ML. Обучение и инференс")

model_types, df_cols_data = get_train_data()

# переменная с текущим типом страниц
if 'models_task' not in st.session_state:
    st.session_state.models_task = 0

# переменная состояния страницы обучения моделей
if 'train_task' not in st.session_state:
    st.session_state.train_task = 0

# переменная состояния страницы предсказания
if 'predict_task' not in st.session_state:
    st.session_state.predict_task = 0

# количество моделей, созданных для обучения
if 'creating_count' not in st.session_state:
    st.session_state.creating_count = 0

# количество моделей, отобранных для сравнения качества
if 'choosing_count' not in st.session_state:
    st.session_state.choosing_count = 0

# переменная состояния страницы со списком моделей
if 'list_task' not in st.session_state:
    st.session_state.list_task = 0

# переменная состояния страницы сравнения качества моделей
if 'choose_task' not in st.session_state:
    st.session_state.choose_task = 0

placeholder = st.empty()

if st.session_state.models_task == 0:
    logging.info('Start page opened')
    start_page(placeholder)
elif st.session_state.models_task == 1:
    logging.info('Train page opened')
    train_page(placeholder, model_types, df_cols_data)
elif st.session_state.models_task == 2:
    logging.info('Predict page opened')
    predict_page(placeholder, df_cols_data)
elif st.session_state.models_task == 3:
    logging.info('Models list page opened')
    list_page(placeholder, df_cols_data)
