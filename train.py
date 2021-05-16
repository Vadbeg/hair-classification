"""Module with training of networks"""

import json
import argparse

import torch

from modules.model.training import train_model
from modules.model.network import FaceNet
from modules.data.dataloader import create_dataloader, get_split_datasets
from modules.data.augs import train_augmentations, valid_augmentations
from modules.utils import load_config, load_exclude_file_paths


def parse_arguments():
    """Parses arguments for CLI"""

    parser = argparse.ArgumentParser(description=f'Starts training script')

    parser.add_argument('--config-path', default='config.ini', type=str,
                        help='Path to config file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    config_path = args.config_path

    config = load_config(config_path=config_path)

    train_images_path = config.get('Data', 'train_images_path')
    train_images_long_hair_folder_name = config.get('Data', 'train_images_long_hair_folder_name')
    train_images_short_hair_folder_name = config.get('Data', 'train_images_short_hair_folder_name')

    weights_dir = config.get('Model', 'weights_dir')
    reports_dir = config.get('Model', 'reports_dir')
    model_type = config.get('Model', 'model_type')
    device = config.get('Model', 'device')
    image_size = tuple(json.loads(config.get('Model', 'image_size')))
    valid_size = config.getfloat('Model', 'valid_size')
    batch_size = config.getint('Model', 'batch_size')
    learning_rate = config.getfloat('Model', 'learning_rate')
    learning_rate_decay_factor = config.getfloat('Model', 'learning_rate_decay_factor')
    num_of_output_nodes = config.getint('Model', 'num_of_output_nodes')
    path_to_exclude_file = config.get('Data', 'path_to_exclude_file')

    exclude_file_paths = None
    if path_to_exclude_file:
        exclude_file_paths = load_exclude_file_paths(exclude_file=path_to_exclude_file)

    train_dataset, valid_dataset = get_split_datasets(
        images_root=train_images_path,
        long_hair_folder=train_images_long_hair_folder_name,
        short_hair_folder=train_images_short_hair_folder_name,
        train_augmentations=train_augmentations,
        valid_augmentations=valid_augmentations,
        image_size=image_size,
        valid_size=valid_size,
        file_paths_to_exclude=exclude_file_paths
    )

    train_dataloader = create_dataloader(dataset=train_dataset, batch_size=batch_size)
    valid_dataloader = create_dataloader(dataset=valid_dataset, batch_size=batch_size)

    model = FaceNet(
        model_type=model_type,
        in_channels=3,
        num_of_output_nodes=num_of_output_nodes,
    )
    arc_margin = None

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )
    loss_func = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=learning_rate_decay_factor,
        mode='min',
        patience=3,
        verbose=True
    )

    train_model(
        model=model,
        num_epochs=50,
        optimizer=optimizer,
        loss_func=loss_func,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        device=device,
        report_dir=reports_dir,
        weights_dir=weights_dir,
        arc_face_module=arc_margin
    )
