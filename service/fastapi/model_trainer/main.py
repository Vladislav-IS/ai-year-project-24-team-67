import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from api.v1.api_route import *


app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


@app.get("/", response_model=List[StatusResponse])
async def root():
    # Реализуйте метод получения информации о статусе сервиса.
    return [StatusResponse(status="App Healty")]


## Реализуйте роутер с префиксом /api/v1/models
app.include_router(router, prefix="/api/v1/models")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)