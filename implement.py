import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score, fbeta_score
from torch import nn, optim
import numpy as np
import hmax
from dataloaders import train_dataloader, test_dataloader
from networks import NNetwork, CNNetwork
from settings import USE_FMINST, USE_HMAX_NETWORK, DEBUG, USE_CNN, DEBUG_EPOCHS_VIEW_IMAGE, RESIZE
from utils import view_classify
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
        confusion_matrix = torch.zeros(40, 40)

        prediction_arr = []
        label_arr = []
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in test_dataloader:
                # HMAX c2 flattened feature vector input
                log_ps = network(images)
                validate_loss += criterion(log_ps, labels)

                # Append to confusion matrix
                max_values, preds = torch.max(log_ps, 1)
                y_pred = preds.view(-1)
                for t, p in zip(labels.view(-1), y_pred):
                    confusion_matrix[t.long(), p.long()] += 1

                prediction_arr.extend(y_pred)
                label_arr.extend(labels)

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
        train_loss = running_loss / len(train_dataloader)
        valid_loss = validate_loss / len(test_dataloader.dataset)

        recall = recall_score(label_arr, prediction_arr, average='macro')
        sklearn_accuracy = accuracy_score(label_arr, prediction_arr)
        precision = precision_score(label_arr, prediction_arr, average='macro')
        f1 = f1_score(label_arr, prediction_arr, average='macro')
        f1_beta = fbeta_score(label_arr, prediction_arr, average='macro', beta=0.5)

        train_losses.append(train_loss)
        validate_losses.append(valid_loss)
        accuracy_data.append(accuracy / len(test_dataloader))

        print("Epoch: {}/{}.. ".format(_, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_dataloader)),
              "Validate Loss: {:.3f}.. ".format(validate_loss / len(test_dataloader)),
              "Accuracy: {:.3f}".format(accuracy / len(test_dataloader)))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(network.state_dict(), 'orl_database_faces.pt')
            print('Recall {:.3f}\n'
                  'Accuracy {:.3f}\n'
                  'Precision {:.3f}\n'
                  'f1 {:.3f}\n'
                  'f1 Beta {:.3f}\n'.format(
                recall,
                sklearn_accuracy,
                precision,
                f1,
                f1_beta,
            ))
            print('Saving confusion matrix ...')
            df = pd.DataFrame(confusion_matrix.numpy())
            df.to_excel('confusion-matrix.xlsx', index=False)
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

#
# from sklearn.metrics import zero_one_score
#
# y_pred = svm.predict(test_samples)
# accuracy = zero_one_score(y_test, y_pred)
# error_rate = 1 - accuracy
