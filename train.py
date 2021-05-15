"""Module with training of networks"""

import json

import torch
import numpy as np
import pandas as pd

from modules.model.training import train_model
from modules.model.network import HerbariumNet
from modules.model.custom_layers import ArcMarginProduct, AddMarginProduct
from modules.model.losses import LabelSmoothingLoss, SoftTripleLoss
from modules.data.dataloader import create_dataloader, split_dataset
from modules.data.augs import train_augmentations, valid_augmentations
from modules.utils import load_config  #, set_seed


if __name__ == '__main__':
    # set_seed(seed_number=27)

    config_path = '/home/vadbeg/Projects/Kaggle/herbarium-2020/config.ini'
    config = load_config(config_path=config_path)

    train_images_path = config.get('Data', 'train_images_folder')
    train_dataframe_path = config.get('Data', 'train_dataframe_path')

    weights_dir = config.get('Model', 'weights_dir')
    reports_dir = config.get('Model', 'reports_dir')
    device = config.get('Model', 'device')
    model_type = config.get('Model', 'model_type')
    image_size = tuple(json.loads(config.get('Model', 'image_size')))
    valid_size = config.getfloat('Model', 'valid_size')
    batch_size = config.getint('Model', 'batch_size')
    learning_rate = config.getfloat('Model', 'learning_rate')
    num_of_output_nodes = config.getint('Model', 'num_of_output_nodes')
    label_smoothing = config.getfloat('Model', 'label_smoothing')

    train_dataframe = pd.read_csv(train_dataframe_path)

    # categories = train_dataframe['category_id'].unique()
    # train_dataframe = train_dataframe.loc[train_dataframe['category_id'].isin(
    #     categories[:10]
    # )]

    # train_dataframe = train_dataframe.sample(frac=0.1)

    train_dataset, valid_dataset = split_dataset(
        dataframe=train_dataframe,
        images_path=train_images_path,
        train_augmentations=train_augmentations,
        valid_augmentations=valid_augmentations,
        image_size=image_size,
        valid_size=valid_size,
    )

    train_dataloader = create_dataloader(dataset=train_dataset, batch_size=batch_size)
    valid_dataloader = create_dataloader(dataset=valid_dataset, batch_size=batch_size)

    model = HerbariumNet(model_type=model_type, pretrained=True,
                         num_of_output_nodes=num_of_output_nodes,
                         get_embeddings=False)
    # arc_margin = AddMarginProduct(in_features=model.n_features, out_features=num_of_output_nodes,
    #                               s=30, m=0.3).to(device=torch.device(device))
    arc_margin = None

    # loss_func = SoftTripleLoss(embedding_size=512, n_class=num_of_output_nodes)
    # optimizer = torch.optim.Adam(params=[{'params': model.parameters()}, {'params': loss_func.parameters()}],
    #                              lr=learning_rate)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = LabelSmoothingLoss(num_classes=num_of_output_nodes, smoothing=label_smoothing)

    train_model(model=model,
                num_epochs=50,
                optimizer=optimizer,
                loss_func=loss_func,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                device=device,
                report_dir=reports_dir,
                weights_dir=weights_dir,
                arc_face_module=arc_margin)

