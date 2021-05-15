"""Module with utils for model training"""

import os
import json
from typing import List, Dict

import torch
import numpy as np
from colorama import Fore


def print_report(metrics: Dict[str, List[float]]):
    """
    Prints report with average metric values to stdout.

    :param metrics: dictionary wuth all metrics
    """

    for metric_name, metrics_values_list in metrics.items():
        print(f'{Fore.RED}{metric_name}: {sum(metrics_values_list) / len(metrics_values_list)}'
              f'{Fore.RESET}')


def save_weights(model: torch.nn.Module, metrics_list, weights_dir: str, file_name: str):
    """
    Save weights to weights_dir

    :param model: model to save
    :param metrics_list: list with dicts of metrics
    :param weights_dir: dir to save weights in
    :param file_name: name of the report file
    """

    if not os.path.exists(weights_dir):
        os.makedirs(name=weights_dir)

    weights_path = os.path.join(weights_dir,
                                f'{file_name}.pt')

    if len(metrics_list) > 1:
        valid_loss_average_list = [float(np.mean(curr_metric['valid_loss'])) for curr_metric in metrics_list]

        valid_loss_list_last = round(valid_loss_average_list[-1], 4)
        valid_loss_list_min = round(min(valid_loss_average_list[:-1]), 4)

        if valid_loss_list_last < valid_loss_list_min:
            torch.save(model, weights_path)

            print(f'{Fore.GREEN}Weights were saved. {valid_loss_list_min} --> {valid_loss_list_last}'
                  f'{Fore.RESET}')


def save_report(report_dir: str, file_name: str, metrics: Dict[str, List[float]], epoch_idx: int):
    """
    Saves report on disk

    :param report_dir: folder to the report
    :param file_name: name of the report file
    :param metrics: metrics for current epoch
    :param epoch_idx: current epoch idx
    """

    if not os.path.exists(report_dir):
        os.makedirs(name=report_dir)

    report_path = os.path.join(report_dir,
                               f'{file_name}.json')

    if os.path.exists(report_path):
        with open(report_path, mode='r', encoding='UTF-8') as file:

            json_loaded: List = json.load(file)

    else:
        json_loaded = []

    metrics = {metric_name: np.mean(metrics_values_list)
               for metric_name, metrics_values_list in metrics.items()}

    metrics = {epoch_idx: metrics}
    json_loaded.append(metrics)

    with open(report_path, mode='w', encoding='UTF-8') as file:
        json.dump(json_loaded, file, indent=4)


def load_model(model_path: str) -> torch.nn.Module:
    """
    Loads model from give file

    :param model_path: .pt file with model
    :return: torch model
    """

    model = torch.load(f=model_path)

    return model
