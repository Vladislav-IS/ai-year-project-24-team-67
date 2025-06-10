import func_timeout
import pickle
import os
from pathlib import Path
from typing import List, Dict, Callable, Any, Iterator
import torch
from torch import nn, optim, jit
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score


class BaselineModel(nn.Module):
    def __init__(self, input_dim = 30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.output(out))
        return out
    

class ModelConstructor(nn.Module):
    def __init__(self, modules_str_dict: Dict[str, Any]):
        super().__init__()

        modules_eval_list = []
        layers_count = modules_str_dict['layers_count']
        for index in range(layers_count):
            layer_type = modules_str_dict[f'layer_{index}']['layer_type']
            layer = f'nn.{layer_type}('
            del modules_str_dict[f'layer_{index}']['layer_type']
            for param, param_val in modules_str_dict[f'layer_{index}'].items():
                layer += f'{param}={param_val},'
            layer += ')'
            modules_eval_list.append(eval(layer))
        self.seq = nn.Sequential(*modules_eval_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DeepLearningTrainer:
    def __init__(self):
        
        # флаг доступности GPU
        self.CUDA_IS_AVAILABLE = torch.cuda.is_available()

        # флаг занятости GPU (True, если на GPU идет обучение модели)
        self.CUDA_IS_BUSY = False

        # список скейлеров для каждой модели
        self.SCALERS_LIST = {}

    def train(self,
              config: Dict[str, Any],
              X: List[List[float]],
              y: List[float],
              model_dir: str) -> Iterator[Dict[str, Any]]:
        '''
        пайплайн обучения DL-модели
        '''   
        try:
            hyperparams = config["hyperparameters"]
            device = hyperparams["device"].lower()
            model_id = config["id"]
            batch_size = hyperparams["batch_size"]
            model_params = hyperparams["model_params"]
            X_train, X_val, y_train, y_val = train_test_split(X, 
                                                              y, 
                                                              shuffle=True, 
                                                              test_size=hyperparams["test_size"])
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
            train_set = TensorDataset(torch.FloatTensor(X_train), y_train_tensor)
            val_set = TensorDataset(torch.FloatTensor(X_val), y_val_tensor)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
       
            if model_params == 'Baseline':
                model = BaselineModel().to(device)
            else:
                model = ModelConstructor(model_params).to(device)
            
            criterion = eval(f'nn.{hyperparams["loss_params"]}()')
            
            if hyperparams["optimizer_params"]["optimizer_type"] == 'Adam':
                beta_1 = hyperparams["optimizer_params"]["beta_1"]
                del hyperparams["optimizer_params"]["beta_1"]
                beta_2 = hyperparams["optimizer_params"]["beta_2"]
                del hyperparams["optimizer_params"]["beta_2"]
                hyperparams["optimizer_params"]['betas'] = (beta_1, beta_2)

            optimizer_str = f'optim.{hyperparams["optimizer_params"]["optimizer_type"]}(model.parameters(),'
            del hyperparams["optimizer_params"]["optimizer_type"]
            for param, param_val in hyperparams["optimizer_params"].items():
                optimizer_str += f'{param}={param_val},'
            optimizer_str += ')'
            optimizer = eval(optimizer_str)

            if hyperparams["metric"] == 'f1':
                metric = f1_score
            elif hyperparams["metric"] == 'accuracy':
                metric = accuracy_score

            for _ in range(hyperparams["epochs_num"]):
                try:
                    loss_and_metric = func_timeout.func_timeout(
                        hyperparams["time_limit"],
                        self.train_epoch,
                        args=(model, optimizer, criterion, metric, train_loader, val_loader, device)
                    )
                    loss_and_metric['id'] = model_id
                    yield loss_and_metric
                except func_timeout.FunctionTimedOut:
                    yield {"id": model_id, "status": "not trained"}
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    return

            model.eval()
            scripted_model = jit.script(model)
            scripted_model.save(f'{model_dir}/{model_id}.pt')
            open(f"{model_dir}/{model_id}", "w").write("NeuralNetwork")
            if not os.path.isdir(f"{model_dir}/scalers"):
                os.mkdir(f"{model_dir}/scalers")
            pickle.dump(scaler, open(f"{model_dir}/scalers/{model_id}_scaler.pkl", "wb"))
            yield {"id": model_id, "model": model, "status": "trained"}
            self.CURRENT_MODEL_ID = model_id
            yield {"id": model_id, "status": "load"}
        except Exception as e:
            print(str(e))
            yield {"id": model_id, "status": "error"}
            if device == 'cuda':
                torch.cuda.empty_cache()
            return


    def train_epoch(self,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    metric: Callable[[List[float], List[float]], List[float]],
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    device: str) -> Dict[str, float]:
        '''
        обучение DL-модели (одна эпоха)
        '''
        train_loss, val_loss = 0, 0
        train_metric, val_metric = 0, 0
        model.train()
        for X_batch, y_batch in train_loader:
            y_batch = y_batch.view(-1, 1).to(device)
            output = model(X_batch.to(device))
            loss = criterion(output, y_batch)
            train_loss += loss.item()
            if output.size(-1) > 1:
                pred = output.argmax(-1)
            else:
                pred = output > 0.5
            train_metric += metric(y_batch, pred.detach().cpu())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_batch = y_batch.view(-1, 1).to(device)
                output = model(X_batch.to(device))
                loss = criterion(output, y_batch.to(device))
                val_loss += loss.item()
                if output.size(-1) > 1:
                    pred = output.argmax(-1)
                else:
                    pred = output > 0.5
                val_metric += metric(y_batch, pred)
        
        train_loss, val_loss = train_loss / len(train_loader), val_loss / len(val_loader)
        train_metric, val_metric = train_metric / len(train_loader), val_metric / len(val_loader)
        return {'train_loss' : train_loss, 
                'val_loss' : val_loss, 
                'train_metric' : train_metric, 
                'val_metric' : val_metric}

    def predict(self, 
                scaler: StandardScaler,
                model: nn.Module, 
                input_data: List[List[float]], 
                batch_size: int,
                device: str):
        '''
        выполнение предсказаний с помощью модели
        '''
        X_test = scaler.transform(input_data)
        test_set = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        result = torch.Tensor([])
        model.to(device)
        model.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                output = model(X_batch[0].to(device))
                if output.size(-1) > 1:
                    output = output.argmax(-1)
                else:
                    output = output > 0.5
                result = torch.cat([result, output.detach().cpu()], dim=0)
        return result.tolist()

    def read_existing_models(self,  
                             model_dir: str,
                             models_list: Dict[str, Any],
                             models_types_list: Dict[str, str]) -> None:
        '''
        чтение ранее обученных моделей из папки
        '''
        for model in Path(model_dir).glob('*.pt'):
            model_id = model.name.replace('.pt', '')
            device = 'cuda' if self.CUDA_IS_AVAILABLE else 'cpu'
            models_list[model_id] = jit.load(f'{model_dir}/{model_id}.pt', map_location=device)
            models_list[model_id].eval()
            models_types_list[model_id] = 'NeuralNetwork'
            self.SCALERS_LIST[model_id] = pickle.load(open(f'{model_dir}/scalers/{model_id}_scaler.pkl', 'rb'))

    def get_optimizers_params(self, optimizer_type: str) -> Dict[str, str]:
        '''
        получение списка доступных параметров
        оптимайзера того или иного типа
        '''
        if optimizer_type == "SGD":
            return {"lr": "float", 
                    "momentum": "float", 
                    "dampening": "float", 
                    "nesterov": "bool", 
                    "weight_decay": "float"}
        if optimizer_type == "Adam":
            return {"lr": "float", 
                    "beta_1": "float", 
                    "beta_2": "float",
                    "weight_decay": "float"}
        return {}
    
    def get_layers_params(self, layer_type: str) -> Dict[str, str]:
        '''
        получение списка доступных параметров
        слоев нейросети того или иного типа
        '''
        if layer_type == "Linear":
            return {"in_features": "int", 
                    "out_features": "int"}
        if layer_type == "Dropout":
            return {"p": "float"}
        return {}