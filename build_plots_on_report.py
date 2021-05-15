"""Module for building plots on base of reports"""

import os
import json
import argparse
from typing import Dict, List
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore

from modules.utils import load_config


def parse_report(report_path: str) -> Dict:
    """
    Parses report from give path

    :param report_path: path to the report
    :return: dict with metric as a key and list of metric values as a value
    """

    if not os.path.exists(report_path):
        raise FileNotFoundError(f'There is no such report!')

    with open(report_path, mode='r', encoding='UTF-8') as file:
        report_json_temp: List[Dict] = json.load(fp=file)

    report_json = dict()
    for curr_report in report_json_temp:
        if len(curr_report) != 1:
            raise ValueError(f'Structure of report if bad. Fix it!')

        report_json.update(curr_report)

    report_epochs = [int(curr_report_epoch) for curr_report_epoch in report_json.keys()]
    report_epochs.sort()

    all_metrics = defaultdict(list)

    for curr_report_epoch in report_epochs:
        curr_metrics: Dict = report_json[str(curr_report_epoch)]

        for metric_name, metric_value in curr_metrics.items():
            all_metrics[metric_name].append(metric_value)

    return all_metrics


def parse_arguments() -> str:
    """Parses arguments for CLI"""

    parser = argparse.ArgumentParser(description=f'Draw plots on base of given report')

    parser.add_argument('--report-name', default='report.json', type=str,
                        help='Name of report file. Is joined with report folder from config')

    args = parser.parse_args()

    return args.report_name


def build_plots(report: Dict[str, List[float]]):
    """
    Build plots on base of reports

    :param report: report with all metrics
    """

    num_of_plot = len(report) // 2

    fig, axs = plt.subplots(1, num_of_plot, figsize=(5 * num_of_plot, 5))
    axs = axs.flatten()

    report_train = {metric_name: metric_values
                    for metric_name, metric_values in report.items() if 'train' in metric_name}

    for curr_ax, (metric_name, metric_values) in zip(axs, report_train.items()):
        metric_common_name = metric_name.split('_')[1]
        curr_ax.set_title(metric_common_name)

        sns.lineplot(x=range(len(metric_values)), y=metric_values, ax=curr_ax)

        metric_valid_name = 'valid_' + metric_common_name

        if report.get(metric_valid_name):
            metric_valid_value = report[metric_valid_name]
            sns.lineplot(x=range(len(metric_valid_value)), y=metric_valid_value, ax=curr_ax)

            curr_ax.legend(title=metric_common_name, labels=['train', 'valid'])

    plt.show()


def print_report(report: Dict[str, List[float]]):
    index = report['valid_loss'].index(min(report['valid_loss']))

    print(f'Best epoch: {Fore.YELLOW}{index}{Fore.RESET}')

    for metric_name, metric_values in report.items():
        print(f'{metric_name}: {Fore.YELLOW}{metric_values[index]:.3f}{Fore.RESET}')


if __name__ == '__main__':
    config_path = '/home/vadbeg/Projects/Kaggle/herbarium-2020/config.ini'
    config = load_config(config_path=config_path)

    reports_folder = config.get('Model', 'reports_dir')

    report_name = parse_arguments()

    report_path = os.path.join(reports_folder, report_name)

    metrics_dict = parse_report(report_path=report_path)

    print_report(report=metrics_dict)
    build_plots(report=metrics_dict)

