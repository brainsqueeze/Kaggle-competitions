from torch.nn import functional as f
import torch
import numpy as np

from lsst_project.src.model import Classifier

from lsst_project.data import utils

import pandas as pd
import json
import csv
import os

from lsst_project.data import config

BATCH_SIZE = 512
root = os.path.dirname(os.path.abspath(__file__))
model_state = torch.load(config.MODEL_PATH + "model.pth")
with open(config.MODEL_PATH + "model.json", "r") as jf:
    params = json.load(jf)
model = Classifier(**params)
model.load_state_dict(model_state)

with open(config.MODEL_PATH + "data.json", "r") as jf:
    d = json.load(jf)
    one_hot_lookup = {int(k): v for k, v in d["classes"].items()}
    max_length = d["max_seq_length"]
    feature_columns = d["columns"]
    mean = pd.Series(d["stats"]["mean"])
    variance = pd.Series(d["stats"]["variance"])
class_lookup = {v: k for k, v in one_hot_lookup.items()}


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
                num_objects = 1
                yield objects_to_return

        if len(objects) > 0:
            yield objects


def pre_process(data, meta_data):
    assert isinstance(data, pd.DataFrame)
    data["unix_time"] = utils.mjd_to_unix_time(data.mjd)
    data = data.set_index('object_id').join(meta_data.set_index('object_id'))

    data["distmod"].fillna(0, inplace=True)
    return data


def batch_generator(data, batch_ids):
    data = data[data.index.isin(batch_ids)]

    # need to sort everything in descending by sequence lengths
    # because that is the format that sequence packing method expects
    group = data.groupby(data.index)
    lengths = group.size()[batch_ids].sort_values(ascending=False)
    sorted_batch_ids = lengths.index

    x = np.array([
        utils.pad_sequence(
            sequence=data[data.index == object_id][feature_columns].values,
            max_sequence_length=max_length
        ) for object_id in sorted_batch_ids
    ])
    seq_tensor = torch.tensor(x, dtype=torch.float32)

    if USE_GPU:
        seq_tensor = seq_tensor.cuda()

    return sorted_batch_ids, seq_tensor, lengths.values


def run_inference(seq_tensor, sequence_lengths, num_classes):
    outputs = model.forward(
        x=seq_tensor,
        sequence_lengths=sequence_lengths,
        max_sequence_length=max_length
    )
    _, index = torch.max(f.softmax(outputs, dim=1), 1)

    y_hat = np.zeros((len(index), num_classes), dtype=int)
    y_hat[np.arange(y_hat.shape[0]), index] = 1
    return y_hat


def run():
    meta_data = utils.load_meta_data(training=False)
    output_columns = ["object_id"] + ["class_{label}".format(label=label) for label in one_hot_lookup.values()]
    first_insert = True
    num_objects_written = 0

    for batch in stream_test_set():
        data = pd.DataFrame(batch)
        data = pre_process(data, meta_data)

        data[feature_columns] = (data[feature_columns] - mean) / variance
        sorted_batch_ids, seq_tensor, seq_lengths = batch_generator(
            data=data,
            batch_ids=data.index.unique()
        )
        predicted_classes = run_inference(
            seq_tensor=seq_tensor,
            sequence_lengths=seq_lengths,
            num_classes=len(class_lookup)
        )
        inference_output = np.hstack((sorted_batch_ids[:, None], predicted_classes))
        output = pd.DataFrame(inference_output, columns=output_columns).sort_values("object_id")

        if first_insert:
            output.to_csv(root + "/output/submission.csv", index=False)
            first_insert = False
        else:
            output.to_csv(root + "/output/submission.csv", mode="a", header=False, index=False)

        num_objects_written += len(output)
        print(num_objects_written, "written to submission output")


if __name__ == '__main__':
    USE_GPU = True
    run()
