import cv2
import os
import numpy as np
from hmax import HMAX
from settings import DATA_DIR

RAW_DIR = os.path.join(DATA_DIR, 'raw')
# Initialize the model with the universal patch set
print('Constructing model')
model = HMAX('./hmax/universal_patch_set.mat')

subject_1 = cv2.imread(os.path.join(RAW_DIR, 's1', '1.pgm'), 0)

batch = np.array([[subject_1]])

import ipdb
ipdb.set_trace()
