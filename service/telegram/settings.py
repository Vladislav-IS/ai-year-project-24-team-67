import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BOT_TOKEN: str
    FASTAPI_URL: str
    ROUTE: str
    LOG_FILE: str
    TIME_INTERVAL: int

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
