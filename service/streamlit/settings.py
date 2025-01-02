from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # URL сервера при локальном запуске Streamlit и FastAPI
    FASTAPI_URL: str = "http://localhost:8000/"

    # URL сервера при запуске Streamlit в Docker
    # FASTAPI_URL: str = "http://fastapi:8000/"

    # URL сервера при запуске на VPS
    # FASTAPI_URL: str =\
    #     'https://ai-year-project-service-qeke.onrender.com/'

    # добавка к URL
    ROUTE: str = "api/model_service/"

    # ссылка на репозиторий проекта
    GITHUB_URL: str =\
        "https://github.com/Vladislav-IS/ai-year-project-24-team-67"
