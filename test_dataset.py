"""Module for dataset testing"""

import argparse
from typing import Optional

import matplotlib.pyplot as plt

from modules.data.dataset import ImageDataset
from modules.utils import load_config, load_exclude_file_paths
from modules.data.utils import tensor_to_numpy
from modules.data.dataloader import get_image_paths_labels


def parse_arguments():
    """Parses arguments for CLI"""

    parser = argparse.ArgumentParser(description=f'Plots item from dataset by given id')

    parser.add_argument('--config-path', default='config.ini', type=str,
                        help='Path to config file')
    parser.add_argument('-i', '--idx', default=0, type=int,
                        help='Index pf item to show')

    args = parser.parse_args()

    return args


def create_dataset(
    images_root: str,
    short_hair_folder: str,
    long_hair_folder: str,
    path_to_exclude_file: Optional[str] = None,
) -> ImageDataset:
    exclude_file_paths = load_exclude_file_paths(exclude_file=path_to_exclude_file)

    class_labels = {
        short_hair_folder: 0,
        long_hair_folder: 1
    }
    image_paths_labels = get_image_paths_labels(
        images_root=images_root,
        short_hair_folder=short_hair_folder,
        long_hair_folder=long_hair_folder,
        class_labels=class_labels,
        file_paths_to_exclude=exclude_file_paths
    )

    dataset = ImageDataset(
        image_paths_labels=image_paths_labels,
    )

    return dataset


if __name__ == '__main__':
    args = parse_arguments()

    config_path = args.config_path
    idx = args.idx

    config = load_config(config_path=config_path)

    train_images_path = config.get('Data', 'train_images_path')
    train_images_long_hair_folder_name = config.get('Data', 'train_images_long_hair_folder_name')
    train_images_short_hair_folder_name = config.get('Data', 'train_images_short_hair_folder_name')
    path_to_exclude_file = config.get('Data', 'path_to_exclude_file')

    dataset = create_dataset(
        images_root=train_images_path,
        short_hair_folder=train_images_short_hair_folder_name,
        long_hair_folder=train_images_long_hair_folder_name,
        path_to_exclude_file=path_to_exclude_file
    )

    dataset_el = dataset[idx]

    image = dataset_el['image']
    image = tensor_to_numpy(image=image)

    fig, axs = plt.subplots(figsize=(12, 8))

    plt.title(dataset_el['label'])
    plt.imshow(image)
    plt.show()
