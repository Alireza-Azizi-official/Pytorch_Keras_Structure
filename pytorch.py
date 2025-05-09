import torch
import torhcmetrics  
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim 
import numpy as np 

x = np.random.random((100, 3))
y = np.random.random((100, 1))

dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
dataloader = DataLoader(dataset, batch_size = 15, shuffle = True)

model = nn.Sequential(
    nn.Linear(3, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1) 
)
loss_criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.95)
metric = torhcmetrics.MeanSquaredError()
model.train()

for epoch in range(10):
    for data in dataloader:
        features, target = data 
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_criterion(predictions, target)
        loss.backward()
        optimizer.step()
    print(f'Epoc {epoch + 1}, Loss: {loss.item()}')
    
model.eval()

with torch.no_grad():
    all_preds = []
    all_targets  = []
    for data in dataloader:
        features, targets = data 
        predictions = model(features)
        all_preds.append(predictions)
        all_targets.append(targets)
        
    all_preds = torch.cat(all_preds, dim = 0)
    all_targets = torch.cat(all_targets, dim = 0)
    mse = metric(all_preds, all_targets)
    print(f'Mean Squared Error on all data : {mse}')
    
torch.save(model.state_dict(), 'model.pth')
new_model = nn.Sequential(
    nn.Linear(3, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)
new_model.load_state_dict(torch.load('model.pth'))


