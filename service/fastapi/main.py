from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import MessageResponse, router
from settings import Settings

settings = Settings()

app = FastAPI(
    title="year_project",
    docs_url="/api/year_project",
    openapi_url="/api/year_project.json",
)

app.include_router(router, prefix="/api/model_service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=MessageResponse)
async def root():
    '''
    коренвой GET-запрос
    '''
    return MessageResponse(message="Ready to work!")
