import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from settings import PROCESSED_DIR, SHUFFLE_BATCH, RESIZE, USE_FMINST, CAPTURE_DIR
from transforms import HaarFaceDetect

capture_images = datasets.ImageFolder(
    os.path.join(CAPTURE_DIR),
    transform=transforms.Compose([
        HaarFaceDetect(),
        transforms.Grayscale(),
        transforms.Scale(RESIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

capture_dataloader = DataLoader(capture_images, batch_size=10)
