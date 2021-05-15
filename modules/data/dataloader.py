"""Module with functions for dataset manipulations"""

import os
import glob
from typing import Tuple, Callable, Dict, Optional, List

import numpy as np
from torch.utils.data import DataLoader

from modules.data.dataset import ImageDataset


def create_dataloader(dataset: ImageDataset, batch_size: int = 32) -> DataLoader:
    """
    Creates dataloader from dataset

    :param dataset: dataset with images
    :param batch_size: size of batches
    :return: resulted dataloader
    """

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True)

    return dataloader


def get_image_paths_labels(
        images_root: str,
        short_hair_folder: str,
        long_hair_folder: str,
        class_labels: Dict[str, int],
        file_paths_to_exclude: Optional[List[str]] = None
) -> List[Tuple[str, int]]:
    """
    Gets image paths and labels from folder with images

    :param images_root: root folder with image folders for each class
    :param short_hair_folder: short hair folder name
    :param long_hair_folder: long hair folder name
    :param class_labels: classes of labels
    :param file_paths_to_exclude: paths to exclude from list with paths
    :return: list with paths and labels
    """

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


def get_split_datasets(
        images_root: str,
        short_hair_folder: str,
        long_hair_folder: str,
        train_augmentations: Optional[Callable] = None,
        valid_augmentations: Optional[Callable] = None,
        file_paths_to_exclude: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (256, 256), valid_size: float = 0.3
) -> Tuple[ImageDataset,
           ImageDataset]:

    """
    Shuffles dataframe and creates train and valid datasets with it

    :param images_root: root folder with image folders for each class
    :param short_hair_folder: short hair folder name
    :param long_hair_folder: long hair folder name
    :param train_augmentations: training augs
    :param valid_augmentations: validation augs
    :param file_paths_to_exclude: paths to exclude from list with paths
    :param image_size: size of image
    :param valid_size: size of validation part
    :return: train_dataset, valid_dataset
    """

    class_labels = {
        short_hair_folder: 0,
        long_hair_folder: 1
    }

    image_paths_labels = get_image_paths_labels(
        images_root=images_root,
        short_hair_folder=short_hair_folder,
        long_hair_folder=long_hair_folder,
        class_labels=class_labels,
        file_paths_to_exclude=file_paths_to_exclude
    )
    np.random.shuffle(image_paths_labels)

    split_idx = round(valid_size * len(image_paths_labels))

    image_paths_labels_train = image_paths_labels[:-split_idx]
    image_paths_labels_valid = image_paths_labels[-split_idx:]

    train_dataset = ImageDataset(image_paths_labels=image_paths_labels_train,
                                 augmentations=train_augmentations,
                                 image_size=image_size)
    valid_dataset = ImageDataset(image_paths_labels=image_paths_labels_valid,
                                 augmentations=valid_augmentations,
                                 image_size=image_size)

    return train_dataset, valid_dataset
