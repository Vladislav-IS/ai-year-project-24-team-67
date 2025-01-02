from typing import Dict, List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_DIR: str = "models"
    LOG_DIR: str = "logs"
    PDF_PATH: str = "content/eda.pdf"
    NUM_CPUS: int = 6
    TARGET_COL: str = "Label"
    NON_FEATURE_COLS: Optional[List[str]] = ["Weight"]
    INDEX_COL: str = "EventId"
    MODEL_TYPES: List[str] = ["LogReg", "SVM",
                              "RandomForest", "GradientBoosting"]
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
    AVAILABLE_SCORINGS: List[str] = ["accuracy", "f1"]
    LOG_CONFIG_PATH: str = "log_config.json"
