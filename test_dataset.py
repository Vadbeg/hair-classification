
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from modules.utils import load_config, load_exclude_file_paths
from modules.data.dataset import ImageDataset
from modules.data.utils import tensor_to_numpy
from modules.data.dataloader import get_split_datasets


if __name__ == '__main__':
    config = load_config(config_path='config.ini')

    train_images_path = config.get('Data', 'train_images_path')
    train_images_long_hair_folder_name = config.get('Data', 'train_images_long_hair_folder_name')
    train_images_short_hair_folder_name = config.get('Data', 'train_images_short_hair_folder_name')

    exclude_file_paths = load_exclude_file_paths(exclude_file='bad_train_files.txt')

    train_dataset, valid_dataset = get_split_datasets(
        images_root=train_images_path,
        long_hair_folder=train_images_long_hair_folder_name,
        short_hair_folder=train_images_short_hair_folder_name
    )

    dataset_el = train_dataset[-1]

    image = dataset_el['image']
    image = tensor_to_numpy(image=image)

    fig, axs = plt.subplots(figsize=(12, 8))

    plt.title(dataset_el['label'])
    plt.imshow(image)
    plt.show()
