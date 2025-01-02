import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
from os import mkdir
from os.path import exists

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # URL сервера при локальном запуске Streamlit и FastAPI
    FASTAPI_URL: str = "http://127.0.0.1:8000/"

    # URL сервера при запуске Streamlit из Docker
    # FASTAPI_URL: str = "http://fastapi:8000/"

    # добавка к URL
    ROUTE: str = "api/model_service/"

    # ссылка на репозиторий проекта
    GITHUB_URL: str =\
        "https://github.com/Vladislav-IS/ai-year-project-24-team-67"


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
