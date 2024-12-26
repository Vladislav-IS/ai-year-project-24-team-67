from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle
from settings.v1 import *
import time
import os


class Services:
    def __init__(self):
        # словарь айди - модель
        self.loaded_models = {} 
        
        # словарь айди - тип модели
        self.loaded_mtypes = {} 
        
        # список модеелй в инференсе
        self.curr_models = [] 
        
        # счетчик активных процессов (1, потому что плюсуем основной процесс сервера)
        self.ACTIVE_PROCESSES = 1 


    def fit(self, X, y, config):
        '''
        обучение модели
        '''
        services.ACTIVE_PROCESSES += 1
        time.sleep(60)
        models = {}
        mtypes = {}
        model_id = config['id']
        mtype = config['ml_model_type']
        hyperparams = config['hyperparameters']
        if mtype == 'linear':
            model = LinearRegression(**hyperparams)
        else:
            model = LogisticRegression(**hyperparams)

        pipeline = Pipeline(steps=[('preprocessor', StandardScaler()),
                                    ('model', model)])
    
        pipeline.fit(X, y)
        if not os.path.exists(settings.MODEL_DIR):
            os.makedirs(settings.MODEL_DIR)
        pickle.dump(pipeline, open(f"{settings.MODEL_DIR}/{model_id}.pkl", "wb"))
        models[model_id] = pipeline
        mtypes[model_id] = mtype
        services.ACTIVE_PROCESSES -= 1
        return models, mtypes


    def find_id(self, model_id):
        '''
        поиск модели в списке всех обученных
        '''
        return self.loaded_models.get(model_id)


    def append_to_global_dict(self, models_and_mtypes):
        '''
        добавление модели в список обученных
        '''
        self.loaded_models.update(models_and_mtypes[0])
        self.loaded_mtypes.update(models_and_mtypes[1])


    def predict(self, X, config):
        overflow = config not in self.curr_models and \
            len(self.curr_models) == settings.NUM_MODELS
        if self.loaded_models.get(config) is None or overflow:
            return None
        y = self.loaded_models[config].predict(X)
        return y

    
    def load(self, config):
        overflow = config in self.curr_models or \
            len(self.curr_models) == settings.NUM_MODELS
        if self.loaded_models.get(config) is None or overflow:
            return False
        self.curr_models.append(config)
        return True 


    def unload(self, model_id):
        print(model_id, self.curr_models)
        if not len(self.curr_models) or model_id not in self.curr_models:
            return False
        self.curr_models.remove(model_id)
        return True


    def list_models(self):
        models = []
        for model_id, mtype in self.loaded_mtypes.items():
            models.append({'id': model_id, 'type': mtype})
        return models


    def remove(self, model_id):
        if self.loaded_models.get(model_id) is None:
            return False
        del self.loaded_models[model_id]
        del self.loaded_mtypes[model_id]
        if model_id in self.curr_models:
            self.curr_models.remove(model_id)
        os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")
        return True


    def remove_all(self):
        self.curr_models.clear()
        ids = list(self.loaded_models.keys())
        for model_id in ids:
            del self.loaded_models[model_id]
            del self.loaded_mtypes[model_id]
            os.remove(f"{settings.MODEL_DIR}/{model_id}.pkl")
        return ids


    def get_status(self):
        return self.curr_models


services = Services()