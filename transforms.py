import numpy as np
from PIL import Image

from utils import get_roi


class HaarFaceDetect:
    """
    Apply opencv haar cascade face detection and return roi cropped image.
    """

    def to_numpy(self, image):
        return np.array(image)

    def from_numpy(self, array):
        return Image.fromarray(array.astype('uint8'), 'RGB')

    def __call__(self, image):
        """

        :param:
            image: As PIL type.
        :return:
            ROI detected image as PIL
        """
        img, roi = get_roi(self.to_numpy(image))
        return self.from_numpy(roi) if roi is not None else image
