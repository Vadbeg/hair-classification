"""Module with configparser"""

from typing import List

import configparser


def load_config(config_path: str) -> configparser.ConfigParser:
    """
    Loads config from give path
    :param config_path: path to the config
    :return: config class
    """

    config = configparser.ConfigParser()

    config.read(filenames=config_path)

    return config


def load_exclude_file_paths(exclude_file: str) -> List[str]:
    exclude_file_paths = list()

    with open(exclude_file, mode='r') as file:
        for curr_line in file.readlines():
            curr_line = curr_line.split(',')[0]

            exclude_file_paths.append(curr_line.strip())

    return exclude_file_paths
