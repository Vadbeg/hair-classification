"""
Module for searching samples with no faces
or more than one face in train and val datasets
"""

import os
import glob

import torch
from cv2 import cv2
from tqdm import tqdm
from facenet_pytorch import MTCNN

from modules.utils import load_config


if __name__ == '__main__':
    config = load_config(config_path='config.ini')

    train_images_path = config.get('Data', 'train_images_path')
    train_images_pattern = os.path.join(train_images_path, '**/*.jpg')

    all_images_names = glob.glob(pathname=train_images_pattern, recursive=True)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    face_detector = MTCNN(device=device)

    bad_training_files = open(file='bad_train_files.txt', mode='a')

    for curr_image_name in tqdm(
            all_images_names,
            postfix=f'Searching for files with more than one face'
    ):
        image = cv2.cvtColor(cv2.imread(curr_image_name), cv2.COLOR_BGR2RGB)

        faces_info = face_detector.detect(img=image)

        if (len(faces_info[1]) != 1) or (faces_info[1][0] is None):
            line = f'{curr_image_name}, {len(faces_info[1])}\n'
            bad_training_files.write(line)

    bad_training_files.close()


