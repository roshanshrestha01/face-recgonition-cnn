import os, sys
import cv2

from matplotlib import pyplot as plt

from settings import HAAR_CASCADE


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_image_roi(image_list):
    if not len(image_list) == 10:
        raise ValueError('List of 10 items is required')
    for idx, im in enumerate(image_list):
        image, roi = get_roi(im)
        plt.subplot(2, 5, idx + 1)
        plt.imshow(image, cmap='gray', interpolation='bicubic')

    plt.show()


def get_roi(img):
    roi = None
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)

    # face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
    # scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
    # - Suppose, the scale factor is 1.03, it means we're using a small
    # step for resizing, i.e. reduce size by 3 %, we increase the chance of a matching
    # size with the model for detection is found, while it's expensive.
    # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # -  This parameter will affect the quality of the detected faces: higher value results in less
    # detections but with higher quality. We're using 1 in the code.
    faces = face_cascade.detectMultiScale(img, 1.3, 1)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = img[y:y + h, x:x + w]

    return img, roi


def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), 0)
                    # if (sz is not None):
                    # im = im.resize(sz, Image.ANTIALIAS)
                    # img, roi = get_roi(im)
                    X.append(im)
                    y.append(c)
                except IOError:
                    print("I/O error")
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1
    return [X, y]
