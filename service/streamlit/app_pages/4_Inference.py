import datetime
import logging
import re

import client_funcs
import pandas as pd
import plotly.express as px
import streamlit as st


def predict_clicked():
    '''
    переход на страницу предсказаний
    '''
    st.session_state.models_task = 1


def list_clicked():
    '''
    переход на страницу списка моделей
    '''
    st.session_state.models_task = 2


def back_clicked():
    '''
    возвращение на стартовую страницу
    '''
    st.session_state.models_task = 0
    st.session_state.list_task = 0
    st.session_state.choosing_count = 0
    st.cache_data.clear()


@st.cache_data
def get_train_data():
    '''
    получение данных для обучения моделей:
    списка типов моделей и списка столбцов
    '''
    response = client_funcs.get_columns()
    df_cols_data = response.json()
    response = client_funcs.get_classic_ml_info()
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
    st.session_state.choosing_count += 1


def delete_clicked():
    '''
    удаление модели на странице обучения или
    сравнения качества
    '''
    st.session_state.choosing_count -= 1


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
        st_cols = st.columns(2)
        st_cols[0].button('Сделать предсказание',
                          on_click=predict_clicked,
                          use_container_width=True)
        st_cols[1].button('Список моделей',
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
        cur_model = response.json()['current']
        baseline_model = response.json()['baseline']
        if cur_model == baseline_model:
            st.info('В качестве текущей модели для инференса установлена \
                    модель-бейзлайн')
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
                res_status, result = make_predictions(
                    st.session_state.predict_csv)
                if res_status:
                    st.info('Предсказания модели представлены ниже \
                            (s - сигнальное событие, b - фоновое событие).')
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
    elif response.status_code == 500:
        st.error('Не найдена ни текущая модель, ни модель-бейзлайн!')
    else:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def set_clicked(model_id):
    '''
    установка модели для инференса
    '''
    response = client_funcs.set_model(model_id)
    if response.status_code != 200:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def unset_clicked():
    '''
    снятие модели с инференса
    '''
    response = client_funcs.unset_model()
    if response.status_code != 200:
        st.error('Текущая модель не найдена')


def remove_model_clicked(model_id):
    '''
    удаление модели
    '''
    response = client_funcs.remove_model(model_id)
    if response.status_code != 200:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def remove_all_clicked():
    '''
    очистка списка моделей
    '''
    response = client_funcs.remove_all()
    if response.status_code != 200:
        st.error(f'Ошибка, сообщение от сервера: {response.content}')


def compare_models_click():
    '''
    переход на страницу сравнения качества моделей
    '''
    st.session_state.list_task = 1


def form_list(models, cur_model, baseline_model):
    '''
    создание виджетов для отображения списка моделей
    '''
    st.session_state.model_ids = []
    for model in models:
        st.divider()
        st.subheader(f'Модель {model['id']}')
        if model['id'] == cur_model:
            st.info('Это текущая модель')
        if model['id'] == baseline_model:
            st.info('Это модель-бейзлайн')
        st.session_state.model_ids.append(model['id'])
        st_cols = st.columns(2)
        st_cols[0].write('Тип:')
        st_cols[1].write(f'{model['type']}')
        for param, value in model['hyperparameters'].items():
            st_cols[0].write(param)
            st_cols[1].write(value)

        if model['id'] == baseline_model and model['id'] != cur_model:
            st_cols = st.columns([1, 2, 1])
            st_cols[1].button('Установить для инференса',
                              key=f'set_{model['id']}',
                              on_click=set_clicked,
                              args=[model['id']],
                              use_container_width=True)
        elif model['id'] != baseline_model:
            st_cols = st.columns(2)
            st_cols[1].button('Удалить',
                              key=f'remove_{model['id']}',
                              on_click=remove_model_clicked,
                              args=[model['id']],
                              use_container_width=True)
            if model['id'] == cur_model:
                st_cols[0].button('Снять с инференса',
                                  key=f'unset_{model['id']}',
                                  on_click=unset_clicked,
                                  use_container_width=True)
            else:
                st_cols[0].button('Установить для инференса',
                                  key=f'set_{model['id']}',
                                  on_click=set_clicked,
                                  args=[model['id']],
                                  use_container_width=True)

    st.divider()
    st_cols = st.columns([1, 2, 1])
    st_cols[1].button('Удалить все модели',
                      on_click=remove_all_clicked,
                      disabled=len(models) == 0,
                      use_container_width=True)
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
            'id': list([f'id: {key}' for key in res.keys()]),
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
    st.info('Здесь вы можете измерить качество одной или нескольких моделей.')
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
    if st.session_state.choose_task == 1:
        ids = set()
        for i in range(st.session_state.choosing_count):
            ids.add(st.session_state[f'choosing_id_{i}'])
        logging.debug(ids)
        response = client_funcs.compare_models({'ids': list(ids)},
                                               st.session_state.predict_csv)
        logging.debug(response.json())
        st.plotly_chart(draw_hist(response.json()['results']),
                        use_container_width=True,
                        theme="streamlit")
        st.session_state.choose_task = 0
        # st.session_state.choosing_count = 0


def list_page(placeholder, cols):
    '''
    страница списка моделей
    '''
    placeholder.empty()
    if st.session_state.list_task == 0:
        response = client_funcs.get_current_model()
        cur_model = ''
        baseline_model = ''
        if response.status_code == 200:
            cur_model = response.json()['current']
            baseline_model = response.json()['baseline']
        response = client_funcs.get_models_list()
        with placeholder.container():
            st_cols = st.columns(3)
            st_cols[1].button("Назад", on_click=back_clicked,
                              use_container_width=True)
            if response.status_code == 200:
                st.header('Список моделей')
                models = response.json()['models']
                form_list(models, cur_model, baseline_model)
            else:
                st.error('Нет обученных моделей')
    elif st.session_state.list_task == 1:
        compare_models_page(placeholder, cols)


logging.info('Classic ML opened')

model_types, df_cols_data = get_train_data()

# переменная с текущим типом страниц
if 'models_task' not in st.session_state:
    st.session_state.models_task = 0

# переменная состояния страницы предсказания
if 'predict_task' not in st.session_state:
    st.session_state.predict_task = 0

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
    logging.info('Predict page opened')
    predict_page(placeholder, df_cols_data)
elif st.session_state.models_task == 2:
    logging.info('Models list page opened')
    list_page(placeholder, df_cols_data)
