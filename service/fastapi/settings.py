from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


def get_columns():
    with open('content/columns.txt') as columns:
        return columns.read().splitlines() 


class Settings(BaseSettings):
    MODEL_DIR: str = 'models'
    PDF_PATH: str = 'content/eda.pdf'
    ZIP_TRAIN_PATH: str = 'content/training.zip'
    ZIP_TEST_PATH: str = 'content/test.zip'
    GITHUB_LINK: str = 'https://github.com/Vladislav-IS/ai-year-project-24-team-67'
    NUM_CPUS: int = 6
    DATAFRAME_COLS: List[str] = get_columns()


settings = Settings()