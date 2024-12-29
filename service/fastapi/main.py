import uvicorn
from fastapi import FastAPI

from router import router, MessageResponse

from os import mkdir
from os.path import dirname, abspath, exists

from fastapi.middleware.cors import CORSMiddleware


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
    return MessageResponse(message="Ready to work!")


if __name__ == "__main__":    
    cur_dir = dirname(abspath(__file__))
    if not exists(f'{cur_dir}/models'):
        mkdir(f'{cur_dir}/models')
    if not exists(f'{cur_dir}/logs'):
        mkdir(f'{cur_dir}/logs')
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=8000, 
                reload=True, 
                log_config=f"{cur_dir}/log.ini")