"""Module with dataset class"""

import os
import glob
from typing import Dict, Callable, Optional, Union, Tuple, List

import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from cv2 import cv2


class ImageDataset(Dataset):
    """Dataset class"""

    def __init__(
            self, images_root: str,
            short_hair_folder: str,
            long_hair_folder: str,
            file_paths_to_exclude: Optional[List[str]] = None,
            augmentations: Optional[Callable] = None,
            image_size: Tuple[int, int] = (256, 256)
    ):
        self.class_labels = {
            short_hair_folder: 0,
            long_hair_folder: 1
        }

        self.image_paths_labels = self.__get_image_paths_labels(
            images_root=images_root,
            short_hair_folder=short_hair_folder,
            long_hair_folder=long_hair_folder,
            class_labels=self.class_labels,
            file_paths_to_exclude=file_paths_to_exclude
        )

        self.__images_root = images_root
        self.__file_paths_to_exclude = file_paths_to_exclude

        self.__augmentations = augmentations
        self.__image_size = image_size

    @staticmethod
    def __get_image_paths_labels(
            images_root: str,
            short_hair_folder: str,
            long_hair_folder: str,
            class_labels: Dict[str, int],
            file_paths_to_exclude: Optional[List[str]] = None
    ) -> List[Tuple[str, int]]:

        short_hair_images_pattern = os.path.join(images_root, short_hair_folder, '*.jpg')
        short_hair_image_paths = set(glob.glob(short_hair_images_pattern))

        long_hair_images_pattern = os.path.join(images_root, long_hair_folder, '*.jpg')
        long_hair_image_paths = set(glob.glob(long_hair_images_pattern))

        short_hair_image_paths_labels = list(zip(
            short_hair_image_paths,
            [class_labels[short_hair_folder]] * len(short_hair_image_paths)
        ))
        long_hair_image_paths_labels = list(zip(
            long_hair_image_paths,
            [class_labels[long_hair_folder]] * len(long_hair_image_paths)
        ))

        image_paths_labels = short_hair_image_paths_labels + long_hair_image_paths_labels

        if file_paths_to_exclude:
            image_paths_labels = [curr_image_path_label
                                  for curr_image_path_label in image_paths_labels
                                  if curr_image_path_label[0] not in file_paths_to_exclude]

        return image_paths_labels

    def __load_image__(self, image_path: str) -> np.ndarray:
        """
        Loads image from disk

        :param image_path: path to the image
        :return: loaded image
        """

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.__image_size)

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

        image_path_label = self.image_paths_labels[idx]

        assert len(image_path_label) == 2, f'Bad image info: {image_path_label}'

        image_path = image_path_label[0]
        image_label = image_path_label[1]

        image = self.__load_image__(image_path=image_path)

        if self.__augmentations:
            image = self.__augmentations(image)['image']
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

        length = len(self.image_paths_labels)

        return length
