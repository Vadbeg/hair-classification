"""Module for CNN model testing"""

import argparse
from typing import Tuple, List, Dict

import torch
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt
from cv2 import cv2
from facenet_pytorch import MTCNN
from albumentations.pytorch.transforms import ToTensorV2

from modules.model.utils import load_model


def parse_arguments():
    """Parses arguments for CLI"""

    parser = argparse.ArgumentParser(
        description=f'Finds all faces on image and checks which kind of hair they have'
    )

    parser.add_argument('--model-path', default='config.ini', type=str,
                        help='Path to model file')
    parser.add_argument('--image-path', default='config.ini', type=str,
                        help='Path to image file')

    args = parser.parse_args()

    return args


def load_image(image_path: str) -> np.ndarray:
    """
    Loads image from disk

    :param image_path: path to the image
    :return: loaded image
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def normalize_image(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Normalizes image

    :param image: image to normalize
    :param image_size: res size of image
    :return: normalized image
    """

    transforms = albu.Compose([
        albu.Normalize(p=1.0),
        albu.Resize(height=image_size[0], width=image_size[1], p=1.0),
        ToTensorV2(p=1.0)
    ])

    image = transforms(image=image)

    return image


def get_result_for_each_face(
        image: np.ndarray,
        model: torch.nn.Module,
        faces_info: List[Dict],
        input_image_size: Tuple[int, int] = (256, 256),
        device: torch.device = torch.device('cuda')
) -> List[Dict]:

    result = list()

    if (len(faces_info[1]) == 1) and (faces_info[1][0] is None):
        return result

    for curr_face_coords in faces_info[0]:
        y_first = int(curr_face_coords[0])
        x_first = int(curr_face_coords[1])
        y_second = int(curr_face_coords[2])
        x_second = int(curr_face_coords[3])

        face_image = image[x_first: x_second, y_first: y_second, :]

        face_image = normalize_image(image=face_image, image_size=input_image_size)['image']
        face_image = face_image.unsqueeze(0)
        face_image = face_image.to(device)

        hair_class = model(face_image)
        hair_class = torch.sigmoid(hair_class)
        hair_class = torch.argmax(hair_class)
        hair_class = hair_class.detach().cpu().numpy()
        hair_class = int(hair_class)

        curr_res = {
            'coords': curr_face_coords,
            'hair_class': hair_class
        }

        result.append(curr_res)

    return result


def plot_bbox_class(
        image: np.ndarray,
        faces_hair_info: List[Dict],
        class_labels: Dict[int, str]
):
    for curr_hair_info in faces_hair_info:
        coords = curr_hair_info['coords']
        text_class = class_labels[curr_hair_info['hair_class']]

        y_first = int(coords[0])
        x_first = int(coords[1])
        y_second = int(coords[2])
        x_second = int(coords[3])

        image = cv2.rectangle(
            image, pt1=(y_first, x_first),
            pt2=(y_second, x_second),
            color=(255, 0, 0),
            thickness=2
        )
        image = cv2.putText(
            image, text=text_class, org=(y_first, x_first - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
            color=(255, 0, 0), thickness=2
        )

    plt.subplots(figsize=(12, 8))
    plt.title(f'Faces')

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    args = parse_arguments()

    model_path = args.model_path
    image_path = args.image_path

    model = load_model(model_path=model_path)

    image = load_image(image_path=image_path)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    face_detector = MTCNN(device=device)

    faces_info = face_detector.detect(img=image)

    class_labels = {
        0: 'short',
        1: 'long'
    }

    result = get_result_for_each_face(
        image=image,
        model=model,
        faces_info=faces_info
    )

    if len(result) == 0:
        print(f'No faces on image')
    else:
        plot_bbox_class(
            image=image,
            faces_hair_info=result,
            class_labels=class_labels
        )
