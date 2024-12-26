import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from router import router, MessageResponse


app = FastAPI(
    title="model_service",
    docs_url="/api/models",
    openapi_url="/api/models.json",
)

app.include_router(router, prefix="/api/models")


@app.get("/", response_model=MessageResponse)
async def root():
    return MessageResponse(message="Ready to work!")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)