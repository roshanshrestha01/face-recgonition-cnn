import torch
from torch import nn, optim
import numpy as np
import hmax
from dataloaders import train_dataloader, validate_dataloader
from networks import NNetwork, CNNetwork
from settings import USE_FMINST, USE_HMAX_NETWORK, DEBUG, USE_CNN, DEBUG_EPOCHS_VIEW_IMAGE, RESIZE
from utils import view_classify, show_batch, pprint_matrix
from matplotlib import pyplot as plt

print('Constructing model')
model = hmax.HMAX('./hmax/universal_patch_set.mat')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running model on', device)
model = model.to(device)
count = 0

epochs = 100

if USE_HMAX_NETWORK:
    network = model
else:
    network = CNNetwork() if USE_CNN else NNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.01)

train_losses, validate_losses, accuracy_data = [], [], []

valid_loss_min = np.Inf
for _ in range(epochs):
    _ += 1
    running_loss = 0
    for images, labels in train_dataloader:
        # HMAX c2 flattened feature vector input
        output = network(images)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        validate_loss = 0
        accuracy = 0
        # confusion_matrix = torch.zeros(40, 40)

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in validate_dataloader:
                # HMAX c2 flattened feature vector input
                log_ps = network(images)
                validate_loss += criterion(log_ps, labels)
                # _, preds = torch.max(log_ps, 1)
                # for t, p in zip(labels.view(-1), preds.view(-1)):
                #     confusion_matrix[t.long(), p.long()] += 1
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
        # pprint_matrix(confusion_matrix.numpy())
        # print('confusion_matrix', confusion_matrix)

        model.train()
        train_loss = running_loss / len(train_dataloader)
        valid_loss = validate_loss / len(validate_dataloader.dataset)
        train_losses.append(train_loss)
        validate_losses.append(valid_loss)
        accuracy_data.append(accuracy / len(validate_dataloader))
        print("Epoch: {}/{}.. ".format(_, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloader)),
              "Validate Loss: {:.3f}.. ".format(validate_loss / len(validate_dataloader)),
              "Accuracy: {:.3f}".format(accuracy / len(validate_dataloader)))

        if valid_loss <= valid_loss_min:
            # print('confusion_matrix_accuracy', confusion_matrix.diag() / confusion_matrix.sum(1))
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(network.state_dict(), 'orl_database_faces.pt')
            valid_loss_min = valid_loss

plt.plot(train_losses, label='Training loss')
plt.plot(validate_losses, label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.legend(frameon=False)
plt.show()
plt.plot(accuracy_data, label='Accuracy')
# plt.yticks([acc * 100 for acc in accuracy_data], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(frameon=False)
plt.show()

# images = model(images)
# log_ps = network(images.reshape(images.shape[0], 1, 8, 400))
