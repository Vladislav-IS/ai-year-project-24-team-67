import pickle
from pathlib import Path
from typing import Any, Dict, List

import func_timeout
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ClassicLearningTrainer:
    def __init__(self):
        pass

    def read_existing_models(self,
                             model_dir: str,
                             models_list: Dict[str, Any],
                             models_types_list: Dict[str, str]) -> None:
        '''
        чтение ранее обученных моделей из папки
        '''
        for model in Path(model_dir).glob('*.pkl'):
            model_id = model.name.replace('.pkl', '')
            models_list[model_id] = pickle.load(
                open(f'{model_dir}/{model.name}', 'rb'))
            models_types_list[model_id] = open(
                f'{model_dir}/{model_id}', 'r').read().strip()

    def train(self,
              config: Dict[str, Any],
              X: List[List[float]],
              y: List[float],
              model_dir: str) -> Dict[str, Any]:
        '''
        обучение модели
        '''
        try:
            model_id = config["id"]
            model_type = config["type"]
            time_limit = config['hyperparameters']['time_limit']
            hyperparams = {}
            for param, value in config["hyperparameters"].items():
                if param != 'time_limit':
                    if isinstance(value, str):
                        hyperparams[param] = None if value == 'None' \
                            else value.lower()
                    else:
                        hyperparams[param] = value
            if model_type == "LogReg":
                model = LogisticRegression(**hyperparams)
            elif model_type == "SVM":
                model = SVC(**hyperparams)
            elif model_type == "RandomForest":
                model = RandomForestClassifier(**hyperparams)
            elif model_type == "GradientBoosting":
                model = GradientBoostingClassifier(**hyperparams)
            pipeline = Pipeline(
                steps=[("preprocessor", StandardScaler()), ("classifier", model)])
            try:
                func_timeout.func_timeout(
                    time_limit, pipeline.fit, args=(X, y))
                pickle.dump(pipeline, open(
                    f"{model_dir}/{model_id}.pkl", "wb"))
                open(f"{model_dir}/{model_id}", "w").write(model_type)
                return {
                    "id": model_id,
                    "model": pipeline,
                    "status": "trained",
                    "type": model_type,
                }
            except func_timeout.FunctionTimedOut:
                return {"id": model_id, "status": "not trained"}
        except Exception as e:
            return {"id": model_id, "status": "error"}

    def get_params(self, model_type: str) -> List[str]:
        '''
        получение списка доступных параметров
        модели того или иного типа
        '''
        if model_type == "LogReg":
            return {"C": "float",
                    "max_iter": "int",
                    "class_weight": "literal/None/Balanced"}
        if model_type == "SVM":
            return {"C": "float",
                    "kernel": "literal/linear/poly/rbf/sigmoid",
                    "degree": "int",
                    "class_weight": "literal/None/balanced"}
        if model_type == "RandomForest":
            return {"n_estimators": "int",
                    "criterion": "literal/gini/entropy/log_loss",
                    "max_depth": "int",
                    "class_weight": "literal/None/balanced"}
        if model_type == "GradientBoosting":
            return {"n_estimators": "int",
                    "criterion": "literal/friedman_mse/squared_error",
                    "max_depth": "int",
                    "learning_rate": "float"}
        return
