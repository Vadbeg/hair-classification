

from modules.utils import load_config, load_exclude_file_paths
from modules.data.dataset import ImageDataset


if __name__ == '__main__':
    config = load_config(config_path='config.ini')

    train_images_path = config.get('Data', 'train_images_path')
    train_images_long_hair_folder_name = config.get('Data', 'train_images_long_hair_folder_name')
    train_images_short_hair_folder_name = config.get('Data', 'train_images_short_hair_folder_name')

    exclude_file_paths = load_exclude_file_paths(exclude_file='bad_train_files.txt')

    image_dataset = ImageDataset(
        images_root=train_images_path,
        short_hair_folder=train_images_short_hair_folder_name,
        long_hair_folder=train_images_long_hair_folder_name,
        # file_paths_to_exclude=exclude_file_paths,
    )

    print(len(image_dataset.image_paths_labels))

    image_dataset = ImageDataset(
        images_root=train_images_path,
        short_hair_folder=train_images_short_hair_folder_name,
        long_hair_folder=train_images_long_hair_folder_name,
        file_paths_to_exclude=exclude_file_paths,
    )

    print(len(image_dataset.image_paths_labels))

    print(len(exclude_file_paths))

