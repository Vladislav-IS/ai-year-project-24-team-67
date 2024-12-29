from typing import Dict, List, Any, Optional, Literal, Annotated

from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel, Field

import asyncio
import concurrent.futures as pool

import os

from settings import settings
from services import services


router = APIRouter()


class MessageResponse(BaseModel):
    message: str
    class Config:
        json_schema_extra = {
            "example": {"message": "Some message text"}
        }


class ModelConfig(BaseModel):
    id: str = Field(min_length=1)
    type: Literal['log_reg', 'svm', 'random_forest', 'boosting']
    hyperparameters: Optional[Dict[str, Any]]
    class Config:
        json_schema_extra = {
            "example": {
                "id": "Some id",
                "type": "svm",
                "hyperparameters": {"some_param": 1.},
                }
        }


class LinkEdaResponse(BaseModel):
    link: str
    class Config:
        json_schema_extra = {
            "example": {"link": "Some link"}
        }


class DataColumnsResponse(BaseModel):
    columns: List[str]
    class Config:
        json_schema_extra = {
            "example": {"columns": ['col1', 'col2']}
        }


class TrainRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    models: List[ModelConfig]
    class Config:
        json_schema_extra = {
            "example": {
                "X": [[1., 0.], [1., 1.]],
                "y": [1., 0,],
                "models": [{
                    "id": "Some id",
                    "type": "log_reg",
                    "hyperparameters": {"C": 1.},
                }]
                }
        } 


class IdResponse(BaseModel):
    id: str = Field(min_length=1)
    status: Literal["load", "unload", "trained", "not trained", "removed"]
    class Config:
        json_schema_extra = {
            "example": {
                "id": "Some id",
                "status": "load"
                }
        }


class RequestError(BaseModel):
    detail: str
    class Config:
       json_schema_extra = {
            "example": {"detail": "HTTPException raised"},
        }
       

class PredictRequest(BaseModel):
    X: List[List[float]]
    class Config:
        json_schema_extra = {
            "example": {"X": [[1., 0.], [1., 1.]]}
        }


class PredictResponse(BaseModel):
    predictions: List[float]
    class Config:
        json_schema_extra = {
            "example" : {"predictions" : [1., 0.]}
        }


class CompareModelsRequest(BaseModel):
    type: Optional[Literal['log_reg', 'svm', 'random_forest', 'boosting']]
    ids: Optional[List[str]]
    X: List[List[float]]
    y: List[float]
    scoring: Literal['accuracy', 'f1']
    class Config:
        json_schema_extra = {
            "example": {
                "type": "log_reg",
                "ids": None,
                "X": [[1., 1.], [0., 1,]],
                "y": [1., 0.],
                "scoring": "f1"
                }
        }


class CompareModelsResponse(BaseModel):
    id: str
    score_value: float
    class Config:
        json_schema_extra = {
            "example": {
                "id": "Some id",
                "score_value": 1.
                }
        }


class ModelsListResponse(BaseModel):
    models: List[ModelConfig]
    class Config:
        json_schema_extra = {
            "example": {
            "models": [{
                "id": "Some id",
                "type": "log_reg",
                "hyperparameters": {"C": 1.}
                }]
            }
        }


@router.get("/get_eda_link", response_model=LinkEdaResponse)
async def get_eda_link():
    return LinkEdaResponse(link=settings.GITHUB_LINK)


@router.get("/get_eda_pdf", response_class=FileResponse)
async def get_eda_pdf():
    headers = {"Content-Disposition": f"attachment; filename={settings.PDF_PATH}"}  
    return FileResponse(settings.PDF_PATH, headers=headers, media_type='application/pdf')


@router.get("/get_columns", response_model=DataColumnsResponse)
async def get_columns():
    return DataColumnsResponse(columns=settings.DATAFRAME_COLS)


@router.post("/train", responses={
    200: {'model': List[IdResponse]},
    500: {'model': RequestError}
})
async def train(request: Annotated[TrainRequest, "one data, many models"]):
    available_cpus = min(settings.NUM_CPUS, os.cpu_count()) - services.ACTIVE_PROCESSES
    train_data = jsonable_encoder(request)
    X = train_data['X']
    y = train_data['y']
    models = train_data['models']
    train_proc_num = len(models)
    if train_proc_num > available_cpus:
        raise HTTPException(status_code=500, detail=f"Too many models to train")
    responses = []
    for model in models:
        mid = model['id']
        if services.find_id(mid):
            raise HTTPException(status_code=500, detail=f"Model '{mid}' is fitted")
    executor = pool.ProcessPoolExecutor(max_workers=train_proc_num)
    services.ACTIVE_PROCESSES += train_proc_num
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(executor, services.fit, X, y, model)
             for model in models]  
    results = await asyncio.gather(*tasks)
    services.ACTIVE_PROCESSES -= train_proc_num
    for models_data in results:
        model_id = models_data["id"]
        status = models_data["status"]
        if status == 'trained':
            services.MODELS_LIST[model_id] = models_data["model"]
            services.MODELS_CONFIG_LIST[model_id] = {
                "type": models_data["type"], 
                "hyperparameters": models_data["hyperparameters"]
                }
        responses.append(IdResponse(id=model_id, status=status))
    return responses


@router.get("/get_current_model", response_model=MessageResponse)
async def get_status_api():
    if services.CURRENT_MODEL_ID is None:
        raise HTTPException(status_code=500, detail="ID is None") 
    return MessageResponse(message=f"{services.CURRENT_MODEL_ID}")
 

@router.post("/set_model/{model_id}", responses={
    201: {'model': IdResponse},
    404: {'model': RequestError}
    })
async def set_model(model_id: Annotated[str, "path-like id"] = Path(min_length=1)):
    if not services.find_id(model_id):
        raise HTTPException(status_code=500, detail="Not found")
    if services.CURRENT_MODEL_ID == model_id:
        raise HTTPException(status_code=500, detail="Already set")
    services.CURRENT_MODEL_ID = model_id
    return IdResponse(id=model_id, status="load")


@router.post("/unset_model", responses = {
    200: {'model': IdResponse},
    500: {'model': RequestError}
})
async def unset_model():
    if services.CURRENT_MODEL_ID is None:
        raise HTTPException(status_code=500, detail="Not found")
    model_id = services.CURRENT_MODEL_ID
    services.CURRENT_MODEL_ID = None
    return IdResponse(id=model_id, status="unload")


@router.post("/predict", responses = {
    200: {'model': PredictResponse},
    500: {'model': RequestError}
    })
async def predict(request: Annotated[PredictRequest, "predict with file"]):
    if services.CURRENT_MODEL_ID is None:
        raise HTTPException(status_code=500, detail=f"Not found")
    X = jsonable_encoder(request)['X']
    preds = services.predict(X, services.CURRENT_MODEL_ID)
    return PredictResponse(predictions=preds)


@router.post("/compare_models", responses = {
    200: {'model': List[CompareModelsResponse]},
    500: {'model': RequestError}
    })
async def compare_models(request: Annotated[CompareModelsRequest, "compare models"]):
    models = jsonable_encoder(request)
    X = models['X']
    y = models['y']
    ids = set(models['ids'])
    if models.get('type') is not None:
        models_res = services.compare_models(X, y, models['scoring'], mtype=models['type'])
        if len(models_res) == 0:
            raise HTTPException(status_code=500, detail='Not found')
    else:
        for model_id in ids:
            if not services.find_id(model_id):
                raise HTTPException(status_code=500, detail='Not found')
        models_res = services.compare_models(X, y, models['scoring'], ids=ids)
    responses = []
    for model in models_res:
        responses.append(CompareModelsResponse(id=model['id'], score_value=model['score_value']))
    return responses


@router.get("/models_list", responses = {
    200: {'model': ModelsListResponse},
    500: {'model': RequestError}
    })
async def models_list():
    if len(services.MODELS_CONFIG_LIST) == 0:
        raise HTTPException(status_code=500, detail='Not found')
    models = []
    for model_id, config in services.MODELS_CONFIG_LIST.items():
        models.append({
            'id': model_id,
            'type': config['type'],
            'hyperparameters': config['hyperparameters']
        })
    return ModelsListResponse(models=models)


@router.delete("/remove/{model_id}", responses = {
    200: {'model': IdResponse},
    404: {'model': RequestError}
})
async def remove(model_id: Annotated[str, "path-like id"] = Path(min_length=1)):
    if not services.find_id(model_id):
        raise HTTPException(status_code=500, detail=f"Not found")
    services.remove(model_id)
    return IdResponse(id=model_id, status='removed')


@router.delete("/remove_all", response_model=List[IdResponse])
async def remove_all_api():
    responses = []
    ids = services.remove_all()
    for model_id in ids:
        responses.append(IdResponse(id=model_id, status='removed'))
    return responses
