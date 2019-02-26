import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from settings import PROCESSED_DIR, SHUFFLE_BATCH, RESIZE
from transforms import HaarFaceDetect

train_images = datasets.ImageFolder(
    os.path.join(PROCESSED_DIR, 'train'),
    transform=transforms.Compose([
        HaarFaceDetect(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

validate_images = datasets.ImageFolder(
    os.path.join(PROCESSED_DIR, 'validate'),
    transform=transforms.Compose([
        HaarFaceDetect(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

test_images = datasets.ImageFolder(
    os.path.join(PROCESSED_DIR, 'test'),
    transform=transforms.Compose([
        HaarFaceDetect(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

train_dataloader = DataLoader(train_images, batch_size=10, shuffle=SHUFFLE_BATCH)
validate_dataloader = DataLoader(validate_images, batch_size=10)
test_dataloader = DataLoader(test_images, batch_size=10)
