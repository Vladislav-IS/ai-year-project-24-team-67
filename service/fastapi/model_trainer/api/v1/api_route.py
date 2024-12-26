from typing import Dict, List, Any

from fastapi import APIRouter, Path, HTTPException
from fastapi.encoders import jsonable_encoder
from http import HTTPStatus

from pydantic import BaseModel, Field
import concurrent.futures as pool

import os

from settings.v1 import *
from services.v1 import *

import asyncio


router = APIRouter()


# определяем типы запросов и ответов
class StatusResponse(BaseModel):
    status: str

    class Config:
        json_schema_extra = {
            "example": {"status": "App healthy"},
        }


class ModelConfig(BaseModel):
    id: str
    ml_model_type: str = Field(default='linear')
    hyperparameters: Dict[str, Any]


class FitRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    config: ModelConfig


class FitResponse(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {"message": "Model 'xxx' trained and saved"},
        }


class LoadRequest(BaseModel):
    id: str


class LoadResponse(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {"message": "Model 'xxx' loaded"},
        }


class ModelListResponse(BaseModel):
    models: List[Dict[str, str]]

    class Config:
        json_schema_extra = {
            "models": [{
                "id": "xxx",
                "type": "linear"
                }],
        }


class PredictRequest(BaseModel):
    id: str
    X: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[float]

    class Config:
        json_schema_extra = {
            "example": {"predictions": [0., 1., 1., 0.]},
        }


class RemoveResponse(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {"message": "Model 'xxx' removed"},
        }


class UnloadRequest(BaseModel):
    id: str


class UnloadResponse(BaseModel):
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {"message": "Model 'xxx' unloaded"},
        }

class HTTPError(BaseModel):
    detail: str

    class Config:
        json_schema_extra = {
            "example": {"detail": "HTTPException raised"},
        }


# API endpoints
@router.post("/fit", responses= {
    200: {'model': List[FitResponse]},
    500: {'model': List[HTTPError]}
})
async def fit_api(request: List[FitRequest]):
    '''
    обучение моделей в отдельных процессах
    '''
    fit_processes = len(request)
    available_cpus = min(settings.NUM_CPUS, os.cpu_count()) - services.ACTIVE_PROCESSES
    if available_cpus < fit_processes:
        raise HTTPException(status_code=500, detail="CPU count is too small")
    msg = []

    input_models = []
    input_ids = []
    for r in request:
        data = jsonable_encoder(r)
        mid = data['config']['id']
        if services.find_id(mid) is not None or mid in input_ids:
                raise HTTPException(status_code=500, detail=f"Model '{mid}' is fitted")
        input_models.append(data)
        input_ids.append(mid)

    executor = pool.ProcessPoolExecutor(max_workers=fit_processes)
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(executor, services.fit, data['X'], data['y'], data['config'])
             for data in input_models]    
    results = await asyncio.gather(*tasks)

    for models_and_mtypes in results:
        services.append_to_global_dict(models_and_mtypes)
    
    for mid in input_ids:
        msg.append(FitResponse(message=f"Model '{mid}' trained and saved"))
    return msg
    

@router.post("/load", responses = {
    201: {'model': List[LoadResponse]},
    404: {'model': HTTPError}
    })
async def load_api(request: LoadRequest):
    '''
    загрузка модели в инференс (не больше 2 штук)
    '''
    model = jsonable_encoder(request)
    model_id = model['id']
    if not services.load(model_id):
        raise HTTPException(status_code=404, detail=f"Cannot load '{model_id}'")
    return [LoadResponse(message=f"Model '{model_id}' loaded")]


@router.get("/get_status", response_model=List[StatusResponse])
async def get_status_api():
    '''
    получение состояния инференса
    '''
    ids = services.get_status()
    if ids is None:
        return [{"status": "Model Status Not Ready"}]  
    return [StatusResponse(status=f"Model Status {', '.join(ids)} Ready")]
 

@router.post("/unload", responses = {
    200: {'model': List[UnloadResponse]},
    500: {'model': HTTPError}
})
async def unload_api(request: UnloadRequest):
    '''
    выгрузка модели из инференса
    '''
    model_id = jsonable_encoder(request)['id']
    if not services.unload(model_id):
        raise HTTPException(status_code=500, detail=f"Cannot unload '{model_id}'")
    return [UnloadResponse(message=f"Model '{model_id}' unloaded")]


@router.post("/predict", responses = {
    201: {'model': List[PredictionResponse]},
    404: {'model': HTTPError}
})
async def predict_api(request: List[PredictRequest]):
    '''
    предсказание с помощью моделей, переданных в запросе (если есть в инференсе)
    '''
    data_array = jsonable_encoder(request)
    for data in data_array:
        preds = services.predict(data['X'], data['id'])
        if preds is None:
            raise HTTPException(status_code=500, detail=f"Cannot predict with '{data['id']}'")
    return [PredictionResponse(predictions=preds)]
    

@router.get("/list_models", response_model=List[ModelListResponse])
async def list_models_api():
    '''
    список всех обученных моеделей
    '''
    return [ModelListResponse(models=services.list_models())]


@router.delete("/remove/{model_id}", responses = {
    200: {'model': List[RemoveResponse]},
    404: {'model': HTTPError}
})
async def remove_api(model_id: str = Path(min_length=1)):
    '''
    удаление модели по айди
    '''
    if not services.remove(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return [RemoveResponse(message=f"Model \'{model_id}\' removed")]


@router.delete("/remove_all", response_model=List[RemoveResponse])
async def remove_all_api():
    '''
    удаление всех моделей
    '''
    msg = []
    ids = services.remove_all()
    for model_id in ids:
        msg.append(LoadResponse(message=f"Model '{model_id}' removed"))
    return msg
