import json
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
from settings import Settings
from streamlit.runtime.uploaded_file_manager import UploadedFile

settings = Settings()


def check_dataset(df: pd.DataFrame,
                  cols_data: Dict[str, Any],
                  mode: str = "train") -> bool:
    '''
    проверка датасета на соответствие
    "эталооным" столбцам и их типам
    '''
    standard_cols = cols_data["columns"].copy()
    df_cols = df.dtypes.to_dict()
    for non_feature_col in cols_data["non_feature"]:
        standard_cols.pop(non_feature_col, None)
        df_cols.pop(non_feature_col, None)
    if mode == "test":
        standard_cols.pop(cols_data["target"], None)
    return df_cols == standard_cols


def get() -> requests.Response:
    '''
    корневой GET-запрос
    '''
    response = requests.get(settings.FASTAPI_URL)
    return response


@st.cache_data
def get_pdf() -> requests.Response:
    '''
    запрос файла PDF
    '''
    response = requests.get(settings.FASTAPI_URL
                            + settings.ROUTE + "get_eda_pdf")
    return response


@st.cache_data
def get_columns() -> requests.Response:
    '''
    запрос "эталонных" столбцов и их типов
    '''
    response = requests.get(settings.FASTAPI_URL
                            + settings.ROUTE + "get_columns")
    return response


@st.cache_data
def get_model_types() -> requests.Response:
    '''
    запрос списка доступных типов моделей
    '''
    response = requests.get(settings.FASTAPI_URL
                            + settings.ROUTE + "get_model_types")
    return response


@st.cache_resource
def train_models(request_list: List[Dict],
                 file: UploadedFile) -> requests.Response:
    '''
    запрос на обучение моделей
    '''
    m = MultipartEncoder(
        fields={
            "models_str": json.dumps(request_list),
            "file": ("data.csv", file, "text/csv"),
        }
    )
    response = requests.post(
        settings.FASTAPI_URL + settings.ROUTE + "train_with_file",
        data=m,
        headers={"Content-Type": m.content_type},
    )
    return response


def get_current_model() -> requests.Response:
    response = requests.get(settings.FASTAPI_URL
                            + settings.ROUTE + "get_current_model")
    return response


@st.cache_data
def predict(file: UploadedFile) -> requests.Response:
    m = MultipartEncoder(
        fields={
            "file": ("data.csv", file, "text/csv"),
        }
    )
    response = requests.post(
        settings.FASTAPI_URL + settings.ROUTE + "predict_with_file",
        data=m,
        headers={"Content-Type": m.content_type},
    )
    return response


def get_models_list() -> requests.Response:
    response = requests.get(settings.FASTAPI_URL
                            + settings.ROUTE + "models_list")
    return response


@st.cache_data
def remove_all() -> requests.Response:
    response = requests.delete(
        settings.FASTAPI_URL + settings.ROUTE + "remove_all")
    return response


@st.cache_data
def remove_model(model_id: str) -> requests.Response:
    response = requests.delete(
        settings.FASTAPI_URL + settings.ROUTE + f"remove/{model_id}"
    )
    return response


@st.cache_data
def set_model(model_id: str) -> requests.Response:
    response = requests.post(
        settings.FASTAPI_URL + settings.ROUTE + f"set_model/{model_id}"
    )
    return response


@st.cache_data
def unset_model() -> requests.Response:
    response = requests.post(settings.FASTAPI_URL
                             + settings.ROUTE + "unset_model")
    return response


@st.cache_data
def compare_models(ids: Dict[str, List[str]],
                   file: UploadedFile) -> requests.Response:
    m = MultipartEncoder(
        fields={"models_str": json.dumps(ids),
                "file": ("data.csv", file, "text/csv")}
    )
    response = requests.post(
        settings.FASTAPI_URL + settings.ROUTE + "compare_models",
        data=m,
        headers={"Content-Type": m.content_type},
    )
    return response
