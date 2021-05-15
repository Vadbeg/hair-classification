"""Module with dataset class"""

import os
from typing import Dict, Callable, Optional, Union, Tuple

import torch
import numpy as np
import pandas as pd
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from cv2 import cv2


class ImageDataset(Dataset):
    """Dataset class"""

    def __init__(self, dataframe: pd.DataFrame, images_path: str,
                 augmentations: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (256, 256)):
        self.dataframe = dataframe

        self.images_path = images_path
        self.augmentations = augmentations

        self.image_size = image_size

    def __load_image__(self, image_path: str) -> np.ndarray:
        """
        Loads image from disk

        :param image_path: path to the image
        :return: loaded image
        """

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image[:, :, ::-1]  # COLOR_BGR2RGB
        image = cv2.resize(image, self.image_size)

        return image

    @staticmethod
    def __normalize_image__(image: np.ndarray) -> np.ndarray:
        """
        Normalizes image

        :param image: image to normalize
        :return: normalized image
        """

        transforms = albu.Compose([
            albu.Normalize(p=1.0),
            ToTensorV2(p=1.0)
        ])

        image = transforms(image=image)

        return image

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, int, torch.Tensor]]:
        """
        Reads dataset item from disk

        :param idx: index of dataset element
        :return:
        """

        dataframe_row = self.dataframe.iloc[idx]

        image_name = dataframe_row['file_name']
        image_label = dataframe_row['category_id']

        image_path = os.path.join(self.images_path, image_name)

        image = self.__load_image__(image_path=image_path)

        if self.augmentations:
            image = self.augmentations(image)['image']
        else:
            image = self.__normalize_image__(image)['image']

        image_label = torch.tensor(image_label)

        dataset_element = {
            'image': image,
            'label': image_label
        }

        return dataset_element

    def __len__(self) -> int:
        """
        Returns length of dataset

        :return: length of dataset
        """

        length = len(self.dataframe)

        return length
