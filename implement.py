import torch
from torch import nn, optim

import hmax
from dataloaders import train_dataloader, test_dataloader, validate_dataloader
from networks import NNetwork, CNNetwork
from settings import USE_FMINST, USE_HMAX_NETWORK, DEBUG, USE_CNN, DEBUG_EPOCHS_VIEW_IMAGE, RESIZE
from utils import view_classify, show_batch
from matplotlib import pyplot as plt

print('Constructing model')
model = hmax.HMAX('./hmax/universal_patch_set.mat')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running model on', device)
model = model.to(device)
count = 0

epochs = 25

if USE_HMAX_NETWORK:
    network = model
else:
    network = CNNetwork() if USE_CNN else NNetwork()

# criterion = nn.NLLLoss()
# optimizer = optim.Adam(network.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.01)

train_losses, validate_losses, accuracy_data = [], [], []
for _ in range(epochs):
    _ += 1
    running_loss = 0
    for images, labels in train_dataloader:
        # images = model(images)
        # output = network(images.reshape(images.shape[0], 1, 8, 400))
        output = network(images)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        validate_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in validate_dataloader:
                # images = model(images)
                # log_ps = network(images.reshape(images.shape[0], 1, 8, 400))
                log_ps = network(images)
                validate_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                if DEBUG and _ in DEBUG_EPOCHS_VIEW_IMAGE:
                    img = images[1]
                    plt.imshow(img[0], cmap='gray')
                    plt.show()
                    s_ps = torch.exp(network(img.reshape(1, 1, RESIZE[0], RESIZE[0])))
                    s_top_p, s_top_class = s_ps.topk(1, dim=1)
                    verion = 'Fashion' if USE_FMINST else 'ORL'
                    view_classify(img, s_ps, verion)

        model.train()
        train_losses.append(running_loss / len(train_dataloader))
        validate_losses.append(validate_loss / len(validate_dataloader))
        accuracy_data.append(accuracy / len(validate_dataloader))
        print("Epoch: {}/{}.. ".format(_, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloader)),
              "Validate Loss: {:.3f}.. ".format(validate_loss / len(validate_dataloader)),
              "Accuracy: {:.3f}".format(accuracy / len(validate_dataloader)))
    # c2 = model(X[:2, :, :, :])
    # s1, c1, s2, c2 = model.get_all_layers(X.to(device))

plt.plot(train_losses, label='Training loss')
plt.plot(validate_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
plt.plot(accuracy_data, label='Accuracy')
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
# view_classify(img, ps, verion)
