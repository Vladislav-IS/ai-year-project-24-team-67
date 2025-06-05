Сохранять через pkl не рекомендуется официальными руководствами PyTorch. Причина проста: структура самой модели может 
меняться при изменении версий PyTorch или изменений в классе модели, что создаст сложности при восстановлении.

Более распространённый и надёжный способ — сохранять состояние модели (weights) с помощью torch.save() и восстанавливать 
его позже с новым экземпляром модели.

## Загрузка state dict (Файл .pt) обратно в новую модель

```python
new_model = FinalNetwork()
new_model.load_state_dict(torch.load('final_dl_weights.pt'))
new_model.eval()
```

```python
class FinalNetwork(nn.Module):
    def __init__(self, input_dim):
        super(FinalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.output(out))
        return out
```


