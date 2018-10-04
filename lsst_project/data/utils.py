from . import config
import os

import pandas as pd


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
