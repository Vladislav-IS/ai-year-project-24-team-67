import streamlit as st
import httpx
from os import mkdir
from os.path import dirname, abspath, exists
import logging
from logging.handlers import RotatingFileHandler
import asyncio


URL = 'http://127.0.0.1:8000/'


async def get():
    async with httpx.AsyncClient() as client:
        response = await client.get(URL)
        return response


async def main():
    st.set_page_config(
        page_title="Intro",
        page_icon="💻"
        )
    st.title('''
             Intro. Предсказание динамики физической системы с \
             помощью нейросетей (годовой проект)
             ''')
    st.markdown('''
                Куратор - Марк Блуменау ([Telegram](https://t.me/markblumenau), [GitHub](https://github.com/markblumenau)).

                Команда:
                - Михаил Мокроносов ([Telegram](https://t.me/idodir), [GitHub](https://github.com/iDodir));
                - Владислав Семенов ([Telegram](https://t.me/Vladislav_iSemenov), [GitHub](https://github.com/Vladislav-IS));
                - Матвей Спиридонов ([Telegram](https://t.me/spiridonovms), [GitHub](https://github.com/matveyspiridonov)).
                ''')
    st.markdown('''
                Цель проекта - провести многоклассовую классификацию событий (частиц) в эксперименте 
                LHCb (детектор на адронном коллайдере). Будут опробованы как classic ML, так и DL подходы.
                ''')
    st.markdown('''
                На странице **📈 EDA** доступен разведочный анализ данных, на странице **🤖 Classic ML** -- обучение и инференс моделей.
                ''')


    response = await get()
    if response.status_code == 200:
        st.info("Связь с сервером установлена.")
    else:
        st.error(f"Ошибка при запросе API: {response.status_code}.")


if __name__ == '__main__':
    
    cur_dir = dirname(abspath(__file__))
    if not exists(f'{cur_dir}/content'):
        mkdir(f'{cur_dir}/content')
    if not exists(f'{cur_dir}/logs'):
        mkdir(f'{cur_dir}/logs')

    logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s",
    handlers=[RotatingFileHandler('logs/streamlit.log', 'a', 1000000, 10)],
    level=logging.INFO
    )

    asyncio.run(main())