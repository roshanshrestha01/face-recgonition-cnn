import os
import cv2
import numpy as np

from utils import read_images, show_image_roi
from settings import RAW_DIR

[x, y] = read_images(RAW_DIR)

show_image_roi(x[0:10])


