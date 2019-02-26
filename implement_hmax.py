import torch

import hmax
from dataloaders import train_dataloader

print('Constructing model')
model = hmax.HMAX('./hmax/universal_patch_set.mat')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running model on', device)
model = model.to(device)
count = 0

for X, y in train_dataloader:
    count += 1
    selected = X[:2, :, :, :]

    # c2 = model(X[:2, :, :, :])
    # s1, c1, s2, c2 = model.get_all_layers(X.to(device))
