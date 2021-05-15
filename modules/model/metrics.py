"""Module with metrics for training"""

import torch
import numpy as np
from sklearn.metrics import f1_score


def get_accuracy(model_prediction: torch.tensor, true_prediction: torch.tensor) -> float:
    """
    Calculates accuracy with model predictions

    Model predictions are the raw model output
    True predictions are the labels

    :param model_prediction: raw model output
    :param true_prediction: labels
    :return: accuracy
    """

    model_prediction: np.ndarray = model_prediction.detach().cpu().numpy()
    true_prediction: np.ndarray = true_prediction.detach().cpu().numpy()

    model_prediction = model_prediction.argmax(axis=1)

    accuracy_number = np.sum(model_prediction == true_prediction) / len(true_prediction)

    return accuracy_number


def get_f1_score(model_prediction: torch.tensor, true_prediction: torch.tensor) -> float:
    """
    Calculates weighted f1_score with model predictions

    Model predictions are the raw model output
    True predictions are the labels

    :param model_prediction: raw model output
    :param true_prediction: labels
    :return: f1 score number
    """

    model_prediction: np.ndarray = model_prediction.detach().cpu().numpy()
    true_prediction: np.ndarray = true_prediction.detach().cpu().numpy()

    model_prediction = model_prediction.argmax(axis=1)

    f1_score_number = f1_score(y_true=true_prediction, y_pred=model_prediction,
                               average='weighted')

    return f1_score_number
