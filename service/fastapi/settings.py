from typing import Dict, List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # папка для сохранения моделей
    MODEL_DIR: str = "models"

    # папка для сохранения логов
    LOG_DIR: str = "logs"

    # PDF-файл c EDA
    PDF_PATH: str = "content/eda.pdf"

    # максимальное число доступных для обучения процессов
    NUM_CPUS: int = 6

    # название целевой переменной
    TARGET_COL: str = "Label"

    # не используемые столбцы датасета
    NON_FEATURE_COLS: Optional[List[str]] = ["Weight"]

    # сотлбец индексов
    INDEX_COL: str = "EventId"

    # список доступных типов моделей
    MODEL_TYPES: List[str] = ["LogReg", "SVM",
                              "RandomForest", "GradientBoosting"]

    # список столбцов датасета с типами
    DATASET_COLS: Dict[str, str] = {
        "EventId": "int64",
        "DER_mass_MMC": "float64",
        "DER_mass_transverse_met_lep": "float64",
        "DER_mass_vis": "float64",
        "DER_pt_h": "float64",
        "DER_deltaeta_jet_jet": "float64",
        "DER_mass_jet_jet": "float64",
        "DER_prodeta_jet_jet": "float64",
        "DER_deltar_tau_lep": "float64",
        "DER_pt_tot": "float64",
        "DER_sum_pt": "float64",
        "DER_pt_ratio_lep_tau": "float64",
        "DER_met_phi_centrality": "float64",
        "DER_lep_eta_centrality": "float64",
        "PRI_tau_pt": "float64",
        "PRI_tau_eta": "float64",
        "PRI_tau_phi": "float64",
        "PRI_lep_pt": "float64",
        "PRI_lep_eta": "float64",
        "PRI_lep_phi": "float64",
        "PRI_met": "float64",
        "PRI_met_phi": "float64",
        "PRI_met_sumet": "float64",
        "PRI_jet_num": "int64",
        "PRI_jet_leading_pt": "float64",
        "PRI_jet_leading_eta": "float64",
        "PRI_jet_leading_phi": "float64",
        "PRI_jet_subleading_pt": "float64",
        "PRI_jet_subleading_eta": "float64",
        "PRI_jet_subleading_phi": "float64",
        "PRI_jet_all_pt": "float64",
        "Weight": "float64",
        "Label": "object",
    }

    # список доступных метрик качества
    AVAILABLE_SCORINGS: List[str] = ["accuracy", "f1"]

    # путь до файла конфигурации логов для сервера uvicorn
    LOG_CONFIG_PATH: str = "log_config.json"
