import os
from typing import Any, Dict, List, Iterator
from sklearn.metrics import accuracy_score, f1_score
import json

from settings import Settings

from classic_learning import ClassicLearningTrainer
from deep_learning import DeepLearningTrainer


settings = Settings()


class TrainStatus:
    def __init__(self, model_id: str):
        self.train_loss = []
        self.val_loss = []
        self.train_metric = []
        self.val_metric = []
        self.id = model_id
        self.status = 'training started'

    def fill_epoch_data(self, train_data: Dict[str, Any]) -> None:
        self.train_loss.append(train_data['train_loss'])
        self.val_loss.append(train_data['val_loss'])
        self.train_metric.append(train_data['train_metric'])
        self.val_metric.append(train_data['val_metric'])

    def complete_filling_data(self, train_data: Dict[str, Any]) -> None:
        self.status = train_data['status']
        if self.status == 'trained':
            self.model = train_data['model']

    def reset(self, model_id: str):
        self.__init__(model_id)


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

        # тренер моделей классического ML
        self.CLASSIC_ML_TRAINER = ClassicLearningTrainer()

        # тренер DL-моделей
        self.DL_TRAINER = DeepLearningTrainer()

        self.TRAIN_STATUS = TrainStatus('')

    def read_existing_models(self) -> None:
        '''
        чтение ранее обученных моделей из папки
        '''
        self.CLASSIC_ML_TRAINER.read_existing_models(settings.MODEL_DIR, 
                                                     self.MODELS_LIST, 
                                                     self.MODELS_TYPES_LIST)
        self.DL_TRAINER.read_existing_models(settings.MODEL_DIR,
                                             self.MODELS_LIST, 
                                             self.MODELS_TYPES_LIST)

    def classic_ml_fit(
        self, X: List[List[float]], y: List[float], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        '''
        обучение классической ML-модели
        '''
        return self.CLASSIC_ML_TRAINER.train(config, X, y, settings.MODEL_DIR)
        
    def dl_fit(self, 
               X: List[List[float]], 
               y: List[float], 
               config: Dict[str, Any]
    ) -> Iterator[str]:
        '''
        обучение DL-модели
        '''
        self.TRAIN_STATUS.reset(config['id'])
        if config['hyperparameters']['device'] == 'CUDA':
            self.DL_TRAINER.CUDA_IS_BUSY = True
        else:
            self.ACTIVE_PROCESSES += 1
        train_loop = self.DL_TRAINER.train(config, X, y, settings.MODEL_DIR)
        for res in train_loop:
            if res.get('status') is not None:
                self.TRAIN_STATUS.complete_filling_data(res)
            else:
                self.TRAIN_STATUS.fill_epoch_data(res)
        if self.TRAIN_STATUS.status == 'load':
            self.MODELS_LIST[self.TRAIN_STATUS.id] = self.TRAIN_STATUS.model
            self.MODELS_TYPES_LIST[self.TRAIN_STATUS.id] = 'NeuralNetwork'
        if config['hyperparameters']['device'] == 'CUDA':
            self.DL_TRAINER.CUDA_IS_BUSY = False
        else:
            self.ACTIVE_PROCESSES -= 1

    def find_id(self, model_id: str) -> bool:
        '''
        поиск модели в списке по ID
        '''
        return self.MODELS_LIST.get(model_id) is not None

    def predict(self, X: List[List[float]], model_id: str) -> List[str]:
        '''
        выполнение предсказаний
        '''
        if self.MODELS_TYPES_LIST[model_id] in settings.MODEL_TYPES:
            preds = self.MODELS_LIST[model_id].predict(X)
        else:
            device = 'cuda' if self.DL_TRAINER.CUDA_IS_AVAILABLE else 'cpu' 
            preds = self.DL_TRAINER.predict(self.DL_TRAINER.SCALERS_LIST[model_id],
                                            self.MODELS_LIST[model_id], 
                                            X, 
                                            settings.BATCH_SIZE, 
                                            device)    
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
                if self.MODELS_TYPES_LIST[model_id] in settings.MODEL_TYPES:
                    score_value = score(y, self.MODELS_LIST[model_id].predict(X))
                else:
                    device = 'cuda' if self.DL_TRAINER.CUDA_IS_AVAILABLE else 'cpu'
                    print(self.DL_TRAINER.SCALERS_LIST)
                    pred = self.DL_TRAINER.predict(self.DL_TRAINER.SCALERS_LIST[model_id],
                                                   self.MODELS_LIST[model_id], 
                                                   X, 
                                                   settings.BATCH_SIZE, 
                                                   device)
                    score_value = score(y, pred)
                result[scoring][model_id] = score_value
        return result

    def remove(self, model_id: str) -> None:
        '''
        удаление модели по ID
        '''
        if self.CURRENT_MODEL_ID == model_id:
            self.CURRENT_MODEL_ID = settings.BASELINE_MODEL_ID
        if self.MODELS_TYPES_LIST[model_id] == 'NeuralNetwork':
            os.remove(f"{settings.MODEL_DIR}/{model_id}.pt")
            os.remove(f"{settings.MODEL_DIR}/scalers/{model_id}_scaler.pkl")
        else:
            os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")
        os.remove(f"{settings.MODEL_DIR}/{model_id}")
        del self.MODELS_LIST[model_id]
        del self.MODELS_TYPES_LIST[model_id]

    def remove_all(self) -> List[str]:
        '''
        очистка списка моделей
        '''
        self.CURRENT_MODEL_ID = settings.BASELINE_MODEL_ID
        ids = list(self.MODELS_LIST.keys())
        for model_id in ids:
            if model_id != settings.BASELINE_MODEL_ID:
                if self.MODELS_TYPES_LIST[model_id] == 'NeuralNetwork':
                    os.remove(f"{settings.MODEL_DIR}/{model_id}.pt")
                    os.remove(f"{settings.MODEL_DIR}/scalers/{model_id}_scaler.pkl")
                else:
                    os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")
                os.remove(f"{settings.MODEL_DIR}/{model_id}")
                del self.MODELS_LIST[model_id]
                del self.MODELS_TYPES_LIST[model_id]
        return ids

    def get_params(self, model_type: str) -> List[str]:
        '''
        получение списка доступных параметров
        модели того или иного типа
        '''
        if model_type != "NeuralNetwork":
            return self.CLASSIC_ML_TRAINER.get_params(model_type)
        return
