from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle
from settings import settings
from typing import List, Dict, Any, Tuple
import threading
import os
import func_timeout


class Services:
    def __init__(self):
        self.MODELS_LIST = {} 
        self.MODELS_CONFIG_LIST = {} 
        self.ACTIVE_PROCESSES = 1 
        self.CURRENT_MODEL_ID = None


    def fit(self, 
            X: List[List[float]], 
            y: List[float], 
            config: Dict[str, Any] = {}) -> Dict[str, Any]:
        self.ACTIVE_PROCESSES += 1
        model_id = config['id']
        mtype = config['type']
        hyperparams = config['hyperparameters']
        if mtype == 'log_reg':
            model = LogisticRegression(**hyperparams)
        elif mtype == 'svm':
            model = SVC(**hyperparams)
        elif mtype == 'random_forest':
            model = RandomForestClassifier(**hyperparams)
        elif mtype == 'boosting':
            model = GradientBoostingClassifier(**hyperparams)
        pipeline = Pipeline(steps=[('preprocessor', StandardScaler()),
                                    ('classifier', model)])    
        try:
            func_timeout.func_timeout(10, pipeline.fit, args=(X, y))
            pickle.dump(pipeline, open(f"{settings.MODEL_DIR}/{model_id}.pkl", "wb"))
            return {
                "id": model_id, 
                "model": pipeline, 
                "status": "trained",
                "type": mtype, 
                "hyperparameters": model.get_params()
                }
        except func_timeout.FunctionTimedOut:
            return {
                "id": model_id, 
                "status": "not trained"
                }


    def find_id(self, model_id: str) -> bool:
        return self.MODELS_LIST.get(model_id) is not None


    def predict(self, X: List[List[float]], model_id: str) -> List[float]:
        preds = self.MODELS_LIST[model_id].predict(X)
        return preds
    

    def compare_models(self, 
                       X: List[List[float]], 
                       y: List[float], 
                       scoring: str, 
                       ids: List[str] = None, mtype: 
                       str = None) -> Dict[str, float]:
        if scoring == 'accuracy':
            score = accuracy_score
        elif scoring == 'f1':
            score = f1_score
        if mtype is not None:
            result = []
            for model_id, config in self.MODELS_CONFIG_LIST.items():
                if config['type'] == mtype:
                    score_value = score(y, self.MODELS_LIST[model_id].predict(X))
                    result.append({'id': model_id, 'score_value': score_value})
        elif ids is not None:
            for model_id in ids:
                score_value = score(y, self.MODELS_LIST[model_id].predict(X))
                result.append({'id': model_id, 'score_value': score_value})
        return result 
    

    def remove(self, model_id) -> None:
        del self.MODELS_LIST[model_id]
        del self.MODELS_CONFIG_LIST[model_id]
        if services.CURRENT_MODEL_ID == model_id:
            services.CURRENT_MODEL_ID = None
        os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")


    def remove_all(self) -> List[str]:
        self.CURRENT_MODEL_ID = None
        ids = list(self.MODELS_LIST.keys())
        for model_id in ids:
            del self.MODELS_LIST[model_id]
            del self.MODELS_CONFIG_LIST[model_id]
            os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")
        return ids


services = Services()