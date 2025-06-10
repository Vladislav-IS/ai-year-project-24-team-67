import logging
from logging.handlers import RotatingFileHandler
from os import mkdir
from os.path import exists

import client_funcs
import streamlit as st
from settings import Settings

settings = Settings()


logging.info("Intro opened")
st.markdown(
    '''
    Это стартовая страница приложения, разработанного в рамках проекта \
        "Предсказание динамики физической системы с \
        помощью нейросетей".
        '''
)
st.markdown(
    """
            Куратор - Марк Блуменау.

            Команда:
            - Владислав Семенов;
            - Матвей Спиридонов.
            """
)
st.markdown(
    "Цель проекта - провести многоклассовую \
            классификацию событий (частиц) в эксперименте \
            LHCb (детектор на адронном коллайдере). Будут \
            опробованы как classic ML, так и DL подходы."
)
st.markdown(
    "На странице **📊 EDA** доступен разведочный анализ данных, \
            на странице **🤖 Classic ML** -- обучение и инференс моделей."
)
st.markdown(
    "Репозиторий проекта доступен по " f"[ссылке]({settings.GITHUB_URL}).")

response = client_funcs.get()
if response.status_code == 200:
    st.info("Соединение с сервером установлено.")
else:
    st.error(
        "Не удалось соедниниться с сервером. "
        f"Код ошибки: {response.status_code}."
    )
