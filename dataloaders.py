import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from settings import PROCESSED_DIR, SHUFFLE_BATCH, RESIZE, USE_FMINST, CAPTURE_DIR
from transforms import HaarFaceDetect, HMAXTransform

train_images = datasets.ImageFolder(
    os.path.join(PROCESSED_DIR, 'train'),
    transform=transforms.Compose([
        # HaarFaceDetect(),
        # HMAXTransform(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

test_images = datasets.ImageFolder(
    os.path.join(PROCESSED_DIR, 'test'),
    transform=transforms.Compose([
        # HaarFaceDetect(),
        # HMAXTransform(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)
#
# test_images = datasets.ImageFolder(
#     os.path.join(PROCESSED_DIR, 'test'),
#     transform=transforms.Compose([
#         # HaarFaceDetect(),
#         # HMAXTransform(),
#         transforms.Grayscale(),
#         transforms.Scale(RESIZE),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x * 255),
#     ])
# )

capture_images = datasets.ImageFolder(
    os.path.join(CAPTURE_DIR),
    transform=transforms.Compose([
        HaarFaceDetect(),
        HMAXTransform(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

train_dataloader = DataLoader(train_images, batch_size=10, shuffle=SHUFFLE_BATCH)
test_dataloader = DataLoader(test_images, batch_size=10)
# test_dataloader = DataLoader(test_images, batch_size=10)
capture_dataloader = DataLoader(capture_images, batch_size=10)

# Fashion MNIST datasets
if USE_FMINST:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    test_dataloader = DataLoader(testset, batch_size=64, shuffle=True)
