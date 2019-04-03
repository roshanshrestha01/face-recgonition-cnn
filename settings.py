import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
CAPTURE_DIR = os.path.join(DATA_DIR, 'capture')

ORL_TRAINED_MODEL = os.path.join(BASE_DIR, 'orl_database_faces.pt')

RESIZE = (48, 48)
HAAR_CASCADE = os.path.join(BASE_DIR, 'cascade', 'haarcascade_frontalface_default.xml')

SHUFFLE_BATCH = True
SHOW_BATCHES = {
    'train': True,
    'validate': False,
    'test': False,
}

DEBUG = False
DEBUG_EPOCHS_VIEW_IMAGE = [10, 65, 97]

USE_FMINST = False
USE_HMAX_NETWORK = False
USE_CNN = True
