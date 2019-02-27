import torch
from torch import nn, optim

import hmax
from dataloaders import train_dataloader, test_dataloader
from networks import NNetwork
from settings import USE_FMINST, USE_HMAX_NETWORK
from utils import view_classify
from matplotlib import pyplot as plt

print('Constructing model')
model = hmax.HMAX('./hmax/universal_patch_set.mat')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running model on', device)
model = model.to(device)
count = 0

epochs = 15

if USE_HMAX_NETWORK:
    network = model
else:
    network = NNetwork()

criterion = nn.NLLLoss()
optimizer = optim.Adam(network.parameters(), lr=0.003)

train_losses, test_losses = [], []
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
        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in test_dataloader:
                log_ps = network(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()
        train_losses.append(running_loss / len(train_dataloader))
        test_losses.append(test_loss / len(test_dataloader))

        print("Epoch: {}/{}.. ".format(_ + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(test_dataloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(test_dataloader)))
    # c2 = model(X[:2, :, :, :])
    # s1, c1, s2, c2 = model.get_all_layers(X.to(device))


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()


# dataiter = iter(test_dataloader)
# images, labels = dataiter.next()
# img = images[1]
#
# ps = torch.exp(network(img))
# top_p, top_class = ps.topk(1, dim=1)
#
# plt.imshow(img[0])
# plt.show()
# print(top_class[:10, :])
# import ipdb
#
# equals = top_class == labels.view((64, 1))
# accuracy = torch.mean(equals.type(torch.FloatTensor))
#
# verion = 'Fashion' if USE_FMINST else 'ORL'
# ipdb.set_trace()
# view_classify(img, ps, verion)
