"""Module with configparser"""

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

