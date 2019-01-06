import os
import argparse
import shutil
import cv2
from settings import PROCESSED_DIR, RAW_DIR
from utils import check_folder

parser = argparse.ArgumentParser(description='Data sample parameters.')

parser.add_argument('ratio', metavar='N', type=int, nargs='+',
                    help='an integer for data separation')

args = parser.parse_args()

is_two_split = len(args.ratio) == 2
is_three_split = len(args.ratio) == 3

validate = 0
if is_two_split:
    train = args.ratio[0]
    test = args.ratio[1]
    data_sets = ['train', 'test']
elif is_three_split:
    train = args.ratio[0]
    validate = args.ratio[1]
    test = args.ratio[2]
    data_sets = ['train', 'validate', 'test']
else:
    raise ValueError('Ratio should be 2 or 3 values i.e. training and testing or training, validation and testing')

if not sum(args.ratio) == 10:
    raise ValueError('Sum of ratio should be equal 10')

if os.path.exists(PROCESSED_DIR):
    shutil.rmtree(PROCESSED_DIR)
check_folder(PROCESSED_DIR)

subjects = []
for root, dirnames, files in os.walk(RAW_DIR):
    for dirname in dirnames:
        subjects.append(dirname)

for subject in subjects:
    for root, _, files in os.walk(os.path.join(RAW_DIR, subject)):
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if is_two_split:
            training_sets = files[:train]
            testing_sets = files[train:]
            data = [training_sets, testing_sets]
        else:
            training_sets = files[:train]
            other_half = files[train:]
            validating_sets = other_half[:validate]
            testing_sets = other_half[validate:]
            data = [training_sets, validating_sets, testing_sets]

        for param in zip(data_sets, data):
            check_folder(os.path.join(PROCESSED_DIR, param[0]))
            check_folder(os.path.join(PROCESSED_DIR, param[0], subject))
            DUMP_DIR = os.path.join(PROCESSED_DIR, param[0], subject)
            for image_filename in param[1]:
                image_path = os.path.join(root, image_filename)
                image = cv2.imread(image_path, 0)
                cv2.imwrite(os.path.join(DUMP_DIR, image_filename), image)
