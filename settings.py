import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DIR = os.path.join(DATA_DIR, 'raw')

SHUFFLE_BATCH = False

PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

HAAR_CASCADE = os.path.join(BASE_DIR, 'cascade', 'haarcascade_frontalface_default.xml')

RESIZE = (128, 128)
