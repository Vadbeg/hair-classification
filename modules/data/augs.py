"""Module with augmentations"""

import numpy as np
from albumentations import (RandomRotate90, GlassBlur,
                            GaussNoise, Compose,
                            Transpose, Flip,
                            Normalize)
from albumentations.pytorch.transforms import ToTensorV2


def train_augmentations(image: np.ndarray) -> np.ndarray:
    """
    Applies train transforms to given image

    :param image: image to transform
    :return: resulted image
    """

    transforms = Compose([
        Transpose(p=0.5),
        Flip(0.5),
        Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

    image = transforms(image=image)

    return image


def valid_augmentations(image: np.ndarray) -> np.ndarray:
    """
    Applies validation transforms to given image

    :param image: image to transform
    :return: resulted image
    """

    transforms = Compose([
        Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

    image = transforms(image=image)

    return image
