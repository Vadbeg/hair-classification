"""Module with functions for dataset manipulations"""

from typing import Tuple, Callable

import numpy as np
import pandas as pd
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


def split_dataset(dataframe: pd.DataFrame, images_path: str,
                  train_augmentations: Callable, valid_augmentations: Callable,
                  image_size: Tuple[int, int] = (256, 256), valid_size: float = 0.3) -> Tuple[ImageDataset,
                                                                                              ImageDataset]:
    """
    Shuffles dataframe and creates train and valid datasets with it

    :param dataframe: dataframe with image_id and label
    :param images_path: path to images folder
    :param train_augmentations: training augs
    :param valid_augmentations: validation augs
    :param image_size: size of image
    :param valid_size: size of validation part
    :return: train_dataset, valid_dataset
    """

    dataframe = dataframe.sample(frac=1)

    split_idx = round(valid_size * len(dataframe))

    dataframe_train = dataframe.iloc[:-split_idx]
    dataframe_valid = dataframe.iloc[-split_idx:]

    train_dataset = ImageDataset(dataframe=dataframe_train, images_path=images_path,
                                 augmentations=train_augmentations,
                                 image_size=image_size)
    valid_dataset = ImageDataset(dataframe=dataframe_valid, images_path=images_path,
                                 augmentations=valid_augmentations,
                                 image_size=image_size)

    return train_dataset, valid_dataset
