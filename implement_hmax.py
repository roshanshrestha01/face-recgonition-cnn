import torch
from torch import nn, optim

import hmax
from dataloaders import train_dataloader
from networks import NNetwork
from utils import view_classify

print('Constructing model')
model = hmax.HMAX('./hmax/universal_patch_set.mat')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running model on', device)
model = model.to(device)
count = 0

epochs = 5
network = NNetwork()

criterion = nn.NLLLoss()
optimizer = optim.Adam(network.parameters(), lr=0.003)

for _ in range(epochs):
    running_loss = 0
    for images, labels in train_dataloader:
        output = network(images)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Training loss: {}".format(running_loss))
    # c2 = model(X[:2, :, :, :])
    # s1, c1, s2, c2 = model.get_all_layers(X.to(device))

dataiter = iter(train_dataloader)
images, labels = dataiter.next()
img = images[0]

ps = torch.exp(network(img))

view_classify(img, ps, 'ORL')
