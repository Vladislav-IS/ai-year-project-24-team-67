import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import func_timeout
from settings import Settings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

settings = Settings()


class Services:
    def __init__(self):

        # словарь с парами "ID модели - объект Pipeline"
        self.MODELS_LIST = {}

        # словарь с парами "ID модели - тип модели"
        self.MODELS_TYPES_LIST = {}

        # число активных процессов (не считая основного процесса)
        self.ACTIVE_PROCESSES = 0

        # ID модели, установленной для инференса
        self.CURRENT_MODEL_ID = settings.BASELINE_MODEL_ID

    def read_existing_models(self) -> None:
        '''
        чтение ранее обученных моделей из папки
        '''
        for model in Path(settings.MODEL_DIR).glob('*.pkl'):
            model_name = f'{settings.MODEL_DIR}/{model.name}'
            model_id = model.name.replace('.pkl', '')
            self.MODELS_LIST[model_id] =\
                pickle.load(open(model_name, 'rb'))
            self.MODELS_TYPES_LIST[model_id] =\
                open(model_name.replace('.pkl', ''), 'r').read().strip()

    def fit(
        self, X: List[List[float]], y: List[float], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        '''
        обучение модели
        '''
        y = y.apply(lambda x: 1 if x == settings.SIGNAL else 0)
        try:
            model_id = config["id"]
            mtype = config["type"]
            hyperparams = {
                param: value
                for param, value in config["hyperparameters"].items()
                if value != ""
            }
            if mtype == "LogReg":
                model = LogisticRegression(**hyperparams)
            elif mtype == "SVM":
                model = SVC(**hyperparams)
            elif mtype == "RandomForest":
                model = RandomForestClassifier(**hyperparams)
            elif mtype == "GradientBoosting":
                model = GradientBoostingClassifier(**hyperparams)
            pipeline = Pipeline(
                steps=[("preprocessor", StandardScaler()),
                       ("classifier", model)]
            )
            try:
                func_timeout.func_timeout(
                    settings.TIME_LIMIT, pipeline.fit, args=(X, y))
                pickle.dump(
                    pipeline, open(
                        f"{settings.MODEL_DIR}/{model_id}.pkl", "wb")
                )
                open(f"{settings.MODEL_DIR}/{model_id}", "w").write(mtype)
                return {
                    "id": model_id,
                    "model": pipeline,
                    "status": "trained",
                    "type": mtype,
                }
            except func_timeout.FunctionTimedOut:
                return {"id": model_id, "status": "not trained"}
        except Exception:
            return {"id": model_id, "status": "error"}

    def find_id(self, model_id: str) -> bool:
        '''
        поиск модели в списке по ID
        '''
        return self.MODELS_LIST.get(model_id) is not None

    def predict(self, X: List[List[float]], model_id: str) -> List[str]:
        '''
        выполнение предсказаний
        '''
        preds = self.MODELS_LIST[model_id].predict(X)
        return [settings.SIGNAL if pred == 1 else settings.BACKGROUND
                for pred in preds]

    def compare_models(
        self, X: List[List[float]], y: List[float], ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        '''
        сравнение моделей по метрикам
        '''
        y = y.apply(lambda x: 1 if x == settings.SIGNAL else 0)
        result = {}
        for scoring in settings.AVAILABLE_SCORINGS:
            result[scoring] = {}
            if scoring == "accuracy":
                score = accuracy_score
            elif scoring == "f1":
                score = f1_score
            for model_id in ids:
                score_value = score(y, self.MODELS_LIST[model_id].predict(X))
                result[scoring][model_id] = score_value
        return result

    def remove(self, model_id: str) -> None:
        '''
        удаление модели по ID
        '''
        del self.MODELS_LIST[model_id]
        del self.MODELS_TYPES_LIST[model_id]
        if self.CURRENT_MODEL_ID == model_id:
            self.CURRENT_MODEL_ID = settings.BASELINE_MODEL_ID
        os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")
        os.remove(f"{settings.MODEL_DIR}/{model_id}")

    def remove_all(self) -> List[str]:
        '''
        очистка списка моделей
        '''
        self.CURRENT_MODEL_ID = Settings.BASELINE_MODEL_ID
        ids = list(self.MODELS_LIST.keys())
        for model_id in ids:
            if model_id != settings.BASELINE_MODEL_ID:
                del self.MODELS_LIST[model_id]
                del self.MODELS_TYPES_LIST[model_id]
                os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")
                os.remove(f"{settings.MODEL_DIR}/{model_id}")
        return ids

    def get_params(self, model_type: str) -> List[str]:
        '''
        получение списка доступных параметров
        модели того или иного типа
        '''
        if model_type == "LogReg":
            return ["C", "max_iter", "fit_intercept", "class_weight"]
        if model_type == "SVM":
            return ["C", "kernel", "degree", "class_weight"]
        if model_type == "RandomForest":
            return ["n_estimators", "criterion", "max_depth", "class_weight"]
        if model_type == "GradientBoosting":
            return ["n_estimators", "criterion", "max_depth", "learning_rate"]
        return
