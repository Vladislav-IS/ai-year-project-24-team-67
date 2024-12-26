from typing import Dict, List, Any, Optional, Literal

from fastapi import APIRouter, Path, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from http import HTTPStatus

from pydantic import BaseModel, Field

import asyncio
import concurrent.futures as pool

import os

from settings import settings
from models import *


router = APIRouter()


class MessageResponse(BaseModel):
    message: str
    class Config:
        json_schema_extra = {
            "example": {"message": "Some message text"}
        }


class ModelConfig(BaseModel):
    id: str = Field(min_length=1)
    type: str = Field(min_length=1, default='logistic')
    hyperparameters: Optional[Dict[str, Any]]
    report: Optional[Dict[str, Any]]
    class Config:
        json_schema_extra = {
            "example": {
                "id": "Some id",
                "type": "Some type",
                "hyperparameters": {"some_param": 1.},
                "report": {"some_metric": 0.99}
                }
        }


class ModelListResponse(BaseModel):
    models: List[ModelConfig]
    class Config:
        json_schema_extra = {
            "example": {
                "models": [{
                    "id": "Some id",
                    "type": "Some type",
                    "hyperparameters": {"some_param": 1.},
                    "report": {"some_metric": 0.99}
                }]
            }
        }


class PredictionsResponse(BaseModel):
    predictions: List[float]
    class Config:
        json_schema_extra = {
            "example": {"predictions": [0., 1., 1., 0.]}
        } 


class BaseEdaResponse(BaseModel):
    link: Optional[str]
    file: Optional[FileResponse]
    plain: Optional[str]


class FileEdaResponse(BaseEdaResponse):
    file: FileResponse 
    class Config:
        json_schema_extra = {
            "example": {"file": FileResponse()}
        } 


class PlainEdaResponse(BaseEdaResponse):
    plaintext: str
    class Config:
        json_schema_extra = {
            "example": {"plain": "Some plain text"}
        } 


class LinkEdaResponse(BaseEdaResponse):
    link: str
    class Config:
        json_schema_extra = {
            "example": {"link": "Some link"}
        } 


class EdaRequest(BaseModel):
    format: Literal['file', 'link', 'plain']
    class Config:
        json_schema_extra = {
            "example": {"format": "Some format"}
        }


class IdRequest(BaseModel):
    id: str = Field(min_length=1)
    class Config:
        json_schema_extra = {
            "example": {"id": "Some id"},
        } 


class BaseFitRequest(BaseModel):
    data: Optional[UploadFile]
    X: Optional[List[List[float]]]
    y: Optional[List[float]]
    training_models: List[ModelConfig]


class FileFitRequest(BaseFitRequest):
    data: UploadFile = File(...)
    class Config:
        json_schema_extra = {
            "example": {
                "data": File(...),
                "training_models": [{
                    "id": "Some id",
                    "type": "Some type",
                    "hyperparameters": {"some_param": 1.},
                    "report": {"some_metric": 0.99}
                }]}
        } 
    

class RawDataFitRequest(BaseFitRequest):
    X: List[List[float]]
    y: List[float]
    class Config:
        json_schema_extra = {
            "example": {
                "X": [[1, 1], [1, 0]],
                "y": [1, 0],
                "training_models": [{
                    "id": "Some id",
                    "type": "Some type",
                    "hyperparameters": {"some_param": 1.},
                    "report": {"some_metric": 0.99}
                }]}
        } 


class HTTPError(BaseModel):
    detail: str
    class Config:
        json_schema_extra = {
            "example": {"detail": "HTTPException raised"},
        }


'''
        Доступные команды:
        /start - начало работы
        /get_eda - просмотр EDA
        /train_model - обучить модель
        /models_list - получить список обученных моделей
        /set_model - установить модель для инференса
        /predict - получить предсказание модели
        /help - справка
        '''

@router.get("/get_eda", response_model=BaseEdaResponse)
async def get_eda(request: EdaRequest):
    eda_request = jsonable_encoder(request)
    match eda_request['format']:
        case 'link':
            return LinkEdaResponse(link=settings.BASE_LINK)
        case 'file':
            headers = {"Content-Disposition": f"inline; filename={settings.EDA_FILE_PATH}"}  
            return FileEdaResponse(file=FileResponse(settings.EDA_FILE_PATH, 
                                                     headers=headers, 
                                                     media_type='application/pdf'))
    raise HTTPException(status_code=500, detail="Error!")


@router.post("/train_models_with_file", responses= {
    200: {'model': List[MessageResponse]},
    500: {'model': HTTPError}
})
async def fit_api(request: List[FileFitRequest]):
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
