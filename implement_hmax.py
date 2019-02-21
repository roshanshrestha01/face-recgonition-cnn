"""
Run the HMAX model on the example images.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
from matplotlib import pyplot as plt

import hmax

# Initialize the model with the universal patch set
from settings import PROCESSED_DIR, SHUFFLE_BATCH, RESIZE
from transforms import HaarFaceDetect
from utils import show_batch

print('Constructing model')
model = hmax.HMAX('./hmax/universal_patch_set.mat')

# A folder with example images
training_images = datasets.ImageFolder(
    os.path.join(PROCESSED_DIR, 'train'),
    transform=transforms.Compose([
        HaarFaceDetect(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

# A dataloader that will run through all example images in one batch
dataloader = DataLoader(training_images, batch_size=10, shuffle=SHUFFLE_BATCH)

# Determine whether there is a compatible GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Run the model on the example images
print('Running model on', device)
model = model.to(device)
for X, y in dataloader:
    # selected = X[:2, :, :, :]
    show_batch(X, y)
    # s1, c1, s2, c2 = model.get_all_layers(X.to(device))

# print('Saving output of all layers to: output.pkl')
# with open('output.pkl', 'wb') as f:
#     pickle.dump(dict(s1=s1, c1=c1, s2=s2, c2=c2), f)
# print('[done]')
