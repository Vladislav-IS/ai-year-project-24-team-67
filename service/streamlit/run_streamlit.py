import streamlit as st
from st_pages import add_page_title, get_nav_from_toml
from settings import Settings
from logging.handlers import RotatingFileHandler
from os import mkdir
from os.path import exists
import logging


def main():
    st.set_page_config(layout="wide")
    nav = get_nav_from_toml(Settings().TOML_FILE)
    pg = st.navigation(nav)
    add_page_title(pg)
    pg.run()


if __name__ == "__main__":

    # создание папки для хранения логов
    if not exists("logs"):
        mkdir("logs")

    # конфигурация параметров логирования
    logging.basicConfig(
        format="[%(asctime)s.%(msecs)03d] %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                filename="logs/streamlit.log",
                mode="a",
                maxBytes=500000,
                backupCount=5,
                delay=True,
            )
        ],
        level=logging.NOTSET,
    )

    main()