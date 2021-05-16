"""Module with augmentations"""

import numpy as np
from albumentations import (RandomRotate90, GlassBlur,
                            ShiftScaleRotate,
                            GaussNoise, Compose,
                            Flip,
                            Normalize)
from albumentations.pytorch.transforms import ToTensorV2


def train_augmentations(image: np.ndarray) -> np.ndarray:
    """
    Applies train transforms to given image

    :param image: image to transform
    :return: resulted image
    """

    transforms = Compose([
        Flip(0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30
        ),
        GaussNoise(p=0.3),
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
