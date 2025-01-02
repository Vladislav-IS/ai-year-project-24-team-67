import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
from os import mkdir
from os.path import abspath, dirname, exists

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FASTAPI_URL: str = "http://127.0.0.1:8000/"
    # FASTAPI_URL: str = "http://fastapi:8000/"
    ROUTE: str = "api/model_service/"
    GITHUB_URL: str =\
        "https://github.com/Vladislav-IS/ai-year-project-24-team-67"


cur_dir = dirname(abspath(__file__))
if not exists(f"{cur_dir}/logs"):
    mkdir(f"{cur_dir}/logs")

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

settings = Settings()
