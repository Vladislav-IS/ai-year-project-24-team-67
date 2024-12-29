import streamlit as st
import pandas as pd
import asyncio
import httpx
import base64


URL = 'http://127.0.0.1:8000/'


def pdf_clicked():
    st.session_state.eda_type = 1


def github_clicked():
    st.session_state.eda_type = 2


def realtime_clicked():
    st.session_state.eda_type = 3


def back_clicked():
    st.session_state.eda_type = 0


async def get_pdf():
    async with httpx.AsyncClient() as client:
        response = await client.get(URL + 'api/model_service/get_eda_file/pdf')
        return response


async def get_link():
    async with httpx.AsyncClient() as client:
        response = await client.get(URL + 'api/model_service/get_eda_link')
        return response
    

async def get_columns():
    async with httpx.AsyncClient() as client:
        response = await client.get(URL + 'api/model_service/get_columns')
        return response


async def main():
    st.set_page_config(page_title='EDA', 
                       page_icon='📊')
    st.title("EDA. Разведочный анализ данных")

    if 'eda_type' not in st.session_state:
        st.session_state.eda_type = 0

    placeholder = st.empty()

    if st.session_state.eda_type == 0:
        with placeholder.container():
            st.write('Выберите тип отображения разведочного анализа данных')
            st.button('Скачать PDF', on_click=pdf_clicked)
            st.button('Получить ссылку на GitHub', on_click=github_clicked)
            st.button('EDA в реальном времени', on_click=realtime_clicked)
    elif st.session_state.eda_type == 1:
        response = await get_pdf()
        pdf = base64.b64encode(response.content).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,\
        {pdf}" width="800" height="1000" type="application/pdf"></iframe>'
        with placeholder.container():
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.button('Назад', on_click=back_clicked)
    elif st.session_state.eda_type == 2:
        response = await get_link()
        link = response.json()['link']
        with placeholder.container():
            st.code(link, language="python")
            st.link_button('Перейти', link)
            st.button('Назад', on_click=back_clicked)
    elif st.session_state.eda_type == 3:
        response = await get_columns()
        columns = response.json()['columns']
        with placeholder.container():
            train_csv = st.file_uploader('Загрузите тренировочный датасет', type=['csv'])
            test_csv = st.file_uploader('Загрузите тестовый датасет', type=['csv'])
            if train_csv is not None or test_csv is not None:
                if test_csv is None:
                    st.error('Загрузите тестовый датасет')
                elif train_csv is None:
                    st.error('Загрузите тренировочный датасет')
                else:
                    df = pd.read_csv(train_csv)
                    st.dataframe(df.head())


if __name__ == '__main__':
    asyncio.run(main())
