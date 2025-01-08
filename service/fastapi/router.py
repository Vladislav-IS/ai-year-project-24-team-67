import asyncio
import concurrent.futures as pool
import json
import os
from typing import Annotated, Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile
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
    type: Literal["LogReg", "SVM", "RandomForest", "GradientBoosting"]
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


# ответ со списком типов моделей и гиперпараметров
class ModelTypesResponse(BaseModel):
    models: Dict[str, List[str]]

    class Config:
        json_schema_extra = {"example": {
            "models": [{"Some type": ["Some param"]}]}}


# ответ с информацией о модели (обучена, удалена и т.п.)
class IdResponse(BaseModel):
    id: str = Field(min_length=1)
    status: Literal["load", "unload", "trained",
                    "not trained", "removed", "error"]

    class Config:
        json_schema_extra = {"example": {"id": "Some id", "status": "load"}}


# ответ в случае ошибки
class RequestError(BaseModel):
    detail: str

    class Config:
        json_schema_extra = {
            "example": {"detail": "HTTPException raised"},
        }


# ответ с предсказаниями
class PredictResponse(BaseModel):
    predictions: List[float]
    index: List[float]
    index_name: str

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [1.0, 0.0],
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


@router.get("/get_model_types", response_model=ModelTypesResponse)
async def get_model_types():
    '''
    запрос списка типов моделей, которые
    можно обучить
    '''
    model_types = {}
    for mtype in settings.MODEL_TYPES:
        model_types[mtype] = services.get_params(mtype)
    return ModelTypesResponse(models=model_types)


@router.post(
    "/train_with_file",
    responses={200: {"model": List[IdResponse]}, 500: {"model": RequestError}},
)
async def train_with_file(
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
    X = df.drop(settings.NON_FEATURE_COLS + [settings.TARGET_COL], axis=1)
    y = df[settings.TARGET_COL]
    executor = pool.ProcessPoolExecutor(max_workers=train_proc_num)
    services.ACTIVE_PROCESSES += train_proc_num
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(executor, services.fit, X, y, dict(model))
        for model in models
    ]
    results = await asyncio.gather(*tasks)
    services.ACTIVE_PROCESSES -= train_proc_num
    for models_data in results:
        model_id = models_data["id"]
        status = models_data["status"]
        if status == "trained":
            services.MODELS_LIST[model_id] = models_data["model"]
            services.MODELS_TYPES_LIST[model_id] = models_data["type"]
        responses.append(IdResponse(id=model_id, status=status))
    return responses


@router.get("/get_current_model", response_model=MessageResponse)
async def get_status_api():
    '''
    получение текущей модели,
    которая установлена для инференса
    '''
    if services.CURRENT_MODEL_ID is None:
        raise HTTPException(
            status_code=500, detail="Current model ID not found")
    return MessageResponse(message=f"{services.CURRENT_MODEL_ID}")


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
    if services.CURRENT_MODEL_ID is None:
        raise HTTPException(status_code=500, detail="Model ID not found")
    model_id = services.CURRENT_MODEL_ID
    services.CURRENT_MODEL_ID = None
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
    if services.CURRENT_MODEL_ID is None:
        raise HTTPException(status_code=500, detail="Model ID not found")
    df = pd.read_csv(file.file, index_col=settings.INDEX_COL)
    preds = services.predict(df, services.CURRENT_MODEL_ID)
    return PredictResponse(
        predictions=preds, index=df.index.values, index_name=settings.INDEX_COL
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
        hyperparams = services.MODELS_LIST[model_id]['classifier'].get_params()
        hyperparams.pop('n_jobs', None)
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
