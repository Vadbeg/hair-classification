"""Module with utils for data"""

import torch
import numpy as np


def numpy_to_tensor(image: np.ndarray) -> torch.tensor:
    """
    Transforms numpy to tensor

    :param image: numpy image
    :return: tensor image
    """

    image = image.transpose([2, 0, 1])
    image = torch.tensor(image)

    return image


def tensor_to_numpy(image: torch.tensor) -> np.ndarray:
    """
    Transforms torch tensor to numpy

    :param image: torch tensor image
    :return: numpy image
    """

    image = np.array(image)
    image = image.transpose([1, 2, 0])

    return image
