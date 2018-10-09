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


def pre_process(data, meta_data):
    assert isinstance(data, pd.DataFrame)
    data["unix_time"] = mjd_to_unix_time(data.mjd)
    data["flux_upper"] = data["flux"] * (1 + data["flux_err"] / 100.)
    data["flux_lower"] = data["flux"] * (1 - data["flux_err"] / 100.)
    data = data.set_index('object_id').join(meta_data.set_index('object_id'))

    data["hostgal_photoz_upper"] = data["hostgal_photoz"] * (1 + data["hostgal_photoz_err"] / 100.)
    data["hostgal_photoz_lower"] = data["hostgal_photoz"] * (1 - data["hostgal_photoz_err"] / 100.)

    data["distmod"].fillna(0, inplace=True)
    return data


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
