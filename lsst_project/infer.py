from torch.nn import functional as f
import torch
import numpy as np

from lsst_project.data import utils

import pandas as pd
import json
import csv

from lsst_project.data import config

BATCH_SIZE = 512
model = torch.load(config.MODEL_PATH + "model.pth")

with open(config.MODEL_PATH + "data.json", "r") as jf:
    d = json.load(jf)
    one_hot_lookup = d["classes"]
    mean = d["stats"]["mean"]
    variance = d["stats"]["variance"]


def number_caster(value):
    try:
        value = int(value)
    except ValueError:
        value = float(value)
    finally:
        value = value
    return value


def stream_test_set():
    previous_object_id = None
    num_objects = 1
    objects = []

    with open(config.DATA_PATH + "test_set.csv", "r") as test_file:
        reader = csv.DictReader(test_file)
        for row in reader:
            if num_objects % BATCH_SIZE != 0:
                row = {k: number_caster(row[k]) for k in row}
                object_id = row["object_id"]

                if previous_object_id is None:
                    previous_object_id = object_id
                    objects.append(row)
                elif previous_object_id == object_id:
                    objects.append(row)
                else:
                    previous_object_id = object_id
                    num_objects += 1
                    objects.append(row)
            else:
                objects_to_return = objects[:]
                objects = []
                yield objects_to_return

        if len(objects) > 0:
            yield objects


def pre_process(data, meta_data):
    assert isinstance(data, pd.DataFrame)
    data["unix_time"] = utils.mjd_to_unix_time(data.mjd)
    data = data.set_index('object_id').join(meta_data.set_index('object_id'))

    data["distmod"].fillna(0, inplace=True)
    return data


def run_inference(seq_tensor, num_classes):
    outputs = model.forward(seq_tensor)
    _, index = torch.max(f.softmax(outputs, dim=1), 1)

    y_hat = np.zeros((len(index), num_classes), dtype=int)
    y_hat[np.arange(y_hat.shape[0]), index] = 1
    return y_hat


def run():
    meta_data = utils.load_meta_data(training=False)
    for batch in stream_test_set():
        data = pd.DataFrame(batch)
        data = pre_process(data, meta_data)


if __name__ == '__main__':
    run()
