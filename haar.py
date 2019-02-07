from settings import RAW_DIR
from utils import read_images, show_image_roi

[x, y] = read_images(RAW_DIR)

# show_image_roi(x[0:10])

for i in range(40):
    start = i * 10
    end = (i + 1) * 10
    show_image_roi(x[start: end], y[start])
