from . import config
import os

import pandas as pd
import numpy as np


def mjd_to_unix_time(mjd_time):
    return (mjd_time - 40587) * 86400


def list_files():
    return os.listdir(config.DATA_PATH)


def load_data(training=True):
    path = config.DATA_PATH + "test_set.csv"
    if training:
        path = config.DATA_PATH + "training_set.csv"
    return pd.read_csv(path)


def load_meta_data(training=True):
    path = config.DATA_PATH + "test_set_metadata.csv"
    if training:
        path = config.DATA_PATH + "training_set_metadata.csv"
    return pd.read_csv(path)


def encode_targets(target_array, col_lookup):
    assert isinstance(target_array, np.ndarray)
    assert isinstance(col_lookup, dict)

    return np.array([col_lookup[val] for val in target_array])


def one_hot_encode(target_array, col_lookup):
    columns = encode_targets(target_array, col_lookup)

    targets = np.zeros((len(target_array), len(col_lookup)), dtype=np.float32)
    targets[np.arange(len(targets)), columns] = 1
    return targets


def pad_sequence(sequence, max_sequence_length):
    """
    Pads individual text sequences to the maximum length
    seen by the model at training time
    :param sequence: list of integer lookup keys for the vocabulary (list)
    :param max_sequence_length: (int)
    :return: padded sequence (ndarray)
    """

    difference = max_sequence_length - sequence.shape[0]
    if difference > 0:
        pad = np.zeros((difference, sequence.shape[1]), dtype=np.float32)
        return np.concatenate((sequence, pad))
    return sequence[:max_sequence_length]
