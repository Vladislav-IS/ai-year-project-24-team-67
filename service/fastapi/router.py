import asyncio
import concurrent.futures as pool
import json
import os
from typing import Annotated, Dict, List, Literal, Optional, Any
import ast

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from services import Services
from settings import Settings

router = APIRouter()

services = Services()
settings = Settings()

# создаем папки для моделей и логов
if not os.path.exists(settings.MODEL_DIR):
    os.mkdir(settings.MODEL_DIR)
else:
    services.read_existing_models()
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)


# универсальный ответ с сообщением
class MessageResponse(BaseModel):
    message: str

    class Config:
        json_schema_extra = {"example": {"message": "Some message text"}}


# конфигурация модели
class ModelConfig(BaseModel):
    id: str = Field(min_length=1)
    type: str
    hyperparameters: Optional[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "Some id",
                "type": "svm",
                "hyperparameters": {"Some param": 1.0},
            }
        }


# ответ со списком столбцов датасета и их типами
class DataColumnsResponse(BaseModel):
    columns: Dict[str, str]
    target: str
    non_feature: Optional[List[str]]

    class Config:
        json_schema_extra = {
            "example": {
                "columns": ["col1", "col2"],
                "target": "col2",
                "non_feature": ["col1"],
            }
        }


# ответ со списком типов классических ML-моделей и гиперпараметров
class ClassicMlInfoResponse(BaseModel):
    models: Dict[str, Dict[str, str]]

    class Config:
        json_schema_extra = {
            "example": {
            "models": {"Some type": {"Some param": "Some param type"}}
            }
        }


# ответ с информацией о модели (обучена, удалена и т.п.)
class IdResponse(BaseModel):
    id: str = Field(min_length=1)
    status: Literal["load", "unload", "training started", "trained",
                    "not trained", "removed", "error"]

    class Config:
        json_schema_extra = {"example": {"id": "Some id", "status": "load"}}


# ответ с расширенной информацией о модели (для глубокого обучения)
class DlIdResponse(IdResponse):
    train_loss: List[float]
    val_loss: List[float]
    train_metric: List[float]
    val_metric: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "Some id", 
                "status": "load",
                "train_loss": [0.1],
                "train_metric": [0.1],
                "val_loss": [0.1],
                "val_metric": [0.1]
                }
            }


# ответ в случае ошибки
class RequestError(BaseModel):
    detail: str

    class Config:
        json_schema_extra = {
            "example": {"detail": "HTTPException raised"},
        }


# ответ с предсказаниями
class PredictResponse(BaseModel):
    predictions: List[str]
    index: List[float]
    index_name: str

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": ["b", "s"],
                "index": [1.0, 2.0],
                "index_name": "Some name",
            }
        }


# запрос на сравнение качества моделей
class CompareModelsRequest(BaseModel):
    ids: List[str]

    class Config:
        json_schema_extra = {"example": {"ids": ["Some id"]}}


# ответ со сравнением качества моделей
class CompareModelsResponse(BaseModel):
    results: Dict[str, Dict[str, float]]

    class Config:
        json_schema_extra = {"example": {"results": {"Some id": 1.0}}}


# ответ со списком обученных моделей
class ModelsListResponse(BaseModel):
    models: List[ModelConfig]

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {"id": "Some id", "type": "log_reg",
                        "hyperparameters": {"C": 1.0}}
                ]
            }
        }


# ответ с ID текущей модели и модели-бейзлайна
class CurrentAndBaselineResponse(BaseModel):
    current: str
    baseline: str

    class Config:
        json_schema_extra = {
            "example": {
                "current": "Some id",
                "baseline": "Some id"
            }
        }


# ответ с параметрами, необходимыми для обучения нейросетей (устройства, лоссы и т.д.)
class DlInfoResponse(BaseModel):
    devices: List[Literal["CUDA", "CPU"]]
    losses: List[Literal["BCELoss", "MSELoss", "CrossEntropyLoss"]]
    layers: Dict[str, Dict[str, str]]
    optimizers: Dict[str, Dict[str, str]]
    metrics: List[str]
    architectures: List[str]

    class Config:
        json_schema_extra = {
            "example" : {
                "devices": ["CPU", "CUDA"],
                "losses": ["Some loss"],
                "metrics": ["Some metric"],
                "optimizers": {"Some optimizer": {"Some param": "Some param type"}},
                "architectures": ["Some architecture"]
            }
        }


@router.get("/get_eda_pdf", response_class=FileResponse)
async def get_eda_pdf():
    '''
    запрос EDA в виде PDF-файла
    '''
    headers = {
        "Content-Disposition": f"attachment; \
               filename={settings.PDF_PATH}"
    }
    return FileResponse(
        settings.PDF_PATH, headers=headers, media_type="application/pdf"
    )


@router.get("/get_columns", response_model=DataColumnsResponse)
async def get_columns():
    '''
    запрос списка колонок датасета с их типами
    '''
    return DataColumnsResponse(
        columns=settings.DATASET_COLS,
        target=settings.TARGET_COL,
        non_feature=settings.NON_FEATURE_COLS,
    )


@router.get("/get_model_types", response_model=ClassicMlInfoResponse)
async def get_classic_ml_info():
    '''
    запрос списка типов классических ML-моделей, которые
    можно обучить
    '''
    model_types = {}
    for mtype in settings.MODEL_TYPES:
        model_types[mtype] = services.get_params(mtype)
    return ClassicMlInfoResponse(models=model_types)


@router.post(
    "/train_classic_ml",
    responses={200: {"model": List[IdResponse]}, 500: {"model": RequestError}},
)
async def train_classic_ml(
    models_str: Annotated[str, 'models list'] = Form(...),
    file: Annotated[UploadFile, 'csv'] = File(...)
):
    '''
    обучение модели по данным из файла:
    models_str - список моделей;
    file - csv-файл с данными
    '''
    models = []
    unique_ids = []
    for model_str in json.loads(models_str):
        model = ModelConfig(
            id=model_str["id"],
            hyperparameters=model_str["hyperparameters"],
            type=model_str["type"],
        )
        if services.find_id(model.id):
            raise HTTPException(
                status_code=500, detail=f"Model {model.id} is already fitted"
            )
        models.append(model)
        unique_ids.append(model_str['id'])
    if len(unique_ids) > len(set(unique_ids)):
        raise HTTPException(
            status_code=500, detail="Found duplicated IDs"
        )
    available_cpus = (
        min(settings.NUM_CPUS, os.cpu_count()) - services.ACTIVE_PROCESSES
    )
    train_proc_num = len(models)
    if train_proc_num > available_cpus:
        raise HTTPException(
            status_code=500, detail="Too many models to train")
    responses = []
    df = pd.read_csv(file.file, index_col=settings.INDEX_COL)
    df_nan = df.replace(-999, np.nan)
    df_notna = df_nan.dropna(subset=settings.NOT_NA_COLS)
    X = df_notna.drop(settings.NON_FEATURE_COLS + [settings.TARGET_COL], axis=1)
    y = df_notna[settings.TARGET_COL].apply(lambda x: 1 if x == settings.SIGNAL else 0)
    executor = pool.ProcessPoolExecutor(max_workers=train_proc_num)
    services.ACTIVE_PROCESSES += train_proc_num
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(executor, services.CLASSIC_ML_TRAINER.train, dict(model), X, y, settings.MODEL_DIR)
        for model in models
    ]
    results = await asyncio.gather(*tasks)
    services.ACTIVE_PROCESSES -= train_proc_num
    last_trained_id = None
    for models_data in results:
        model_id = models_data["id"]
        status = models_data["status"]
        if status == "trained":
            services.MODELS_LIST[model_id] = models_data["model"]
            services.MODELS_TYPES_LIST[model_id] = models_data["type"]
            last_trained_id = model_id
        responses.append(IdResponse(id=model_id, status=status))
    if last_trained_id is not None:
        services.CURRENT_MODEL_ID = last_trained_id
        responses.append(IdResponse(id=last_trained_id, status='load'))
    return responses


@router.post(
    "/train_dl",
    responses={200: {"model": IdResponse}, 500: {"model": RequestError}},
)
async def train_dl(
    backgroundTasks: BackgroundTasks,
    model_str: Annotated[str, 'pytorch model data'] = Form(...),
    file: Annotated[UploadFile, 'csv'] = File(...)
):
    '''
    обучение модели по данным из файла:
    model_str - параметры DL-модели;
    file - csv-файл с данными
    '''
    model_dict = json.loads(model_str)
    model = ModelConfig(
        id=model_dict["id"],
        type=model_dict["type"],
        hyperparameters=model_dict["hyperparameters"]
        )
    if services.find_id(model.id):
        raise HTTPException(
            status_code=500, detail=f"Model {model.id} is already fitted"
        )
    available_cpus = (
        min(settings.NUM_CPUS, os.cpu_count()) - services.ACTIVE_PROCESSES
    )
    train_on_gpu = model.hyperparameters['device'] == 'CUDA' and not services.DL_TRAINER.CUDA_IS_BUSY
    train_on_cpu = model.hyperparameters['device'] == 'CPU' and available_cpus >= 1
    if not (train_on_cpu or train_on_gpu):
        raise HTTPException(
            status_code=500, detail="Too many models to train")
    df = pd.read_csv(file.file, index_col=settings.INDEX_COL)
    df_nan = df.replace(-999, np.nan)
    df_notna = df_nan.dropna(subset=settings.NOT_NA_COLS)
    X = df_notna.drop(settings.NON_FEATURE_COLS + [settings.TARGET_COL], axis=1)
    y = df_notna[settings.TARGET_COL].apply(lambda x: 1 if x == settings.SIGNAL else 0)
    backgroundTasks.add_task(services.dl_fit, X, y, dict(model))
    return IdResponse(id=model.id, status='training started')


@router.get("/get_dl_status", response_model=DlIdResponse)
async def get_dl_status():
    '''
    получение информации о статусе обучения
    '''
    return DlIdResponse(
        id=services.TRAIN_STATUS.id,
        train_loss=services.TRAIN_STATUS.train_loss,
        val_loss=services.TRAIN_STATUS.val_loss,
        train_metric=services.TRAIN_STATUS.train_metric,
        val_metric=services.TRAIN_STATUS.val_metric,
        status=services.TRAIN_STATUS.status
    )


@router.get("/get_current_model", response_model=CurrentAndBaselineResponse)
async def get_status_api():
    '''
    получение текущей модели,
    которая установлена для инференса
    '''
    if services.CURRENT_MODEL_ID is None and\
            settings.BASELINE_MODEL_ID is None:
        raise HTTPException(
            status_code=500, detail="Current and baseline model ID not found")
    return CurrentAndBaselineResponse(current=services.CURRENT_MODEL_ID,
                                      baseline=settings.BASELINE_MODEL_ID)


@router.post(
    "/set_model/{model_id}",
    responses={200: {"model": IdResponse}, 404: {"model": RequestError}},
)
async def set_model(model_id:
                    Annotated[str, "path-like id"] = Path(min_length=1)):
    '''
    установка модели для инференса:
    model_id - ID модели в виде path-параметра
    '''
    if not services.find_id(model_id):
        raise HTTPException(status_code=500, detail="Model ID not found")
    if services.CURRENT_MODEL_ID == model_id:
        raise HTTPException(
            status_code=500, detail="Model ID is already fitted")
    services.CURRENT_MODEL_ID = model_id
    return IdResponse(id=model_id, status="load")


@router.post(
    "/unset_model",
    responses={200: {"model": IdResponse}, 500: {"model": RequestError}}
)
async def unset_model():
    '''
    снятие текущей модели с инференса
    '''
    if services.CURRENT_MODEL_ID == settings.BASELINE_MODEL_ID:
        raise HTTPException(
            status_code=500, detail="Cannot unset baseline model")
    model_id = services.CURRENT_MODEL_ID
    services.CURRENT_MODEL_ID = settings.BASELINE_MODEL_ID
    return IdResponse(id=model_id, status="unload")


@router.post(
    "/predict_with_file",
    responses={200: {"model": PredictResponse}, 500: {"model": RequestError}},
)
async def predict(file: Annotated[UploadFile, 'csv'] = File(...)):
    '''
    выполнение предсказаний по файлу с данными:
    file - csv-файл с данными
    '''
    df = pd.read_csv(file.file, index_col=settings.INDEX_COL)
    df_nan = df.replace(-999, np.nan)
    df_notna = df_nan.dropna(subset=settings.NOT_NA_COLS)
    preds = services.predict(df_notna, services.CURRENT_MODEL_ID)
    return PredictResponse(
        predictions=preds, index=df_notna.index.values, index_name=settings.INDEX_COL
    )


@router.post(
    "/compare_models",
    responses={200: {"model": CompareModelsResponse},
               500: {"model": RequestError}},
)
async def compare_models(
    models_str: Annotated[str, 'models list'] = Form(...),
    file: Annotated[UploadFile, 'csv'] = File(...)
):
    '''
    сравнение моделей с заданными ID по метрикам:
    models_str - список ID моделей;
    file - csv-файл, по которому выполняется сравнение
    '''
    models = CompareModelsRequest(ids=json.loads(models_str)["ids"])
    df = pd.read_csv(file.file, index_col=settings.INDEX_COL)
    for col in settings.NON_FEATURE_COLS:
        if col in df.columns:
            df = df.drop(col, axis=1)
    X = df.drop(settings.TARGET_COL, axis=1)
    y = df[settings.TARGET_COL]
    models_res = services.compare_models(X, y, models.ids)
    return CompareModelsResponse(results=models_res)


@router.get(
    "/models_list",
    responses={200: {"model": ModelsListResponse},
               500: {"model": RequestError}},
)
async def models_list():
    '''
    запрос списка моделей
    '''
    if len(services.MODELS_TYPES_LIST) == 0:
        raise HTTPException(status_code=500, detail="Models list not found")
    models = []
    for model_id, model_type in services.MODELS_TYPES_LIST.items():
        if model_type in settings.MODEL_TYPES:
            hyperparams = services.MODELS_LIST[model_id]['classifier'].get_params()
            hyperparams = {param: value for param, value in hyperparams.items()
                           if param in services.get_params(model_type)}
        else:
            hyperparams = {}
        models.append(
            {
                "id": model_id,
                "type": model_type,
                "hyperparameters": hyperparams,
            }
        )
    return ModelsListResponse(models=models)


@router.delete(
    "/remove/{model_id}",
    responses={200: {"model": IdResponse}, 500: {"model": RequestError}},
)
async def remove(model_id: Annotated[str, "path-like id"] =
                 Path(min_length=1)):
    '''
    удаление модели:
    model_id - ID модели в виде path-параметра
    '''
    if not services.find_id(model_id):
        raise HTTPException(status_code=500, detail="Model ID not found")
    if model_id == settings.BASELINE_MODEL_ID:
        raise HTTPException(status_code=500, detail="Cannot remove baseline")
    services.remove(model_id)
    return IdResponse(id=model_id, status="removed")


@router.delete("/remove_all", responses={
    200: {'model': List[IdResponse]},
    500: {'model': RequestError}
})
async def remove_all_api():
    '''
    очистка списка моделей
    '''
    if len(services.MODELS_TYPES_LIST) == 0:
        raise HTTPException(status_code=500, detail="Models list not found")
    responses = []
    ids = services.remove_all()
    for model_id in ids:
        responses.append(IdResponse(id=model_id, status="removed"))
    return responses


@router.get("/get_dl_info", response_model=DlInfoResponse)
async def dl_info():
    '''
    запрос параметров, необходимых для обучения нейросетей
    '''
    devices = ['CPU']
    optimizers = {}
    layers = {}
    if services.DL_TRAINER.CUDA_IS_AVAILABLE:
        devices.append('CUDA')
    for optimizer_type in settings.AVAILABLE_OPTIMIZERS:
        optimizers[optimizer_type] = services.DL_TRAINER.get_optimizers_params(optimizer_type)
    for layer_type in settings.AVAILABLE_LAYERS:
        layers[layer_type] = services.DL_TRAINER.get_layers_params(layer_type)
    return DlInfoResponse(devices=devices, 
                          optimizers=optimizers, 
                          layers=layers,
                          losses=settings.AVAILABLE_LOSSES,
                          metrics=settings.AVAILABLE_SCORINGS,
                          architectures=settings.AVAILABLE_ARCHITECTURES)
