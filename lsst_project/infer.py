from torch.nn import functional as f
import torch
import numpy as np

from lsst_project.src.model import LstmCnnClassifier

from lsst_project.data import utils

import pandas as pd
import json
import csv
import os

from lsst_project.data import config

BATCH_SIZE = 512
USE_GPU = True
root = os.path.dirname(os.path.abspath(__file__))

model_state = torch.load(config.MODEL_PATH + "model.pth")
with open(config.MODEL_PATH + "model.json", "r") as jf:
    params = json.load(jf)
model = LstmCnnClassifier(**params)
model.load_state_dict(model_state)
if USE_GPU:
    model = model.cuda()
    BATCH_SIZE *= 2

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
            row = {k: number_caster(row[k]) for k in row}
            object_id = row["object_id"]

            if previous_object_id is None:
                previous_object_id = object_id
            elif object_id != previous_object_id:
                previous_object_id = object_id
                num_objects += 1

            # one over the batch size limit
            if num_objects == BATCH_SIZE + 1:
                objects_to_return = objects[:]
                objects = [row]
                num_objects = 1
                yield objects_to_return
            else:
                objects.append(row)
        if len(objects) > 0:
            yield objects


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
    seq_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=False)

    if USE_GPU:
        seq_tensor = seq_tensor.cuda()

    return sorted_batch_ids, seq_tensor, lengths.values


def run_inference(seq_tensor, sequence_lengths):
    outputs = model.forward(
        x=seq_tensor,
        sequence_lengths=sequence_lengths,
        max_sequence_length=max_length
    )
    y_hat = f.softmax(outputs, dim=1)
    # need probability outputs
    return y_hat.detach().cpu().numpy().astype(np.float32)


def run():
    meta_data = utils.load_meta_data(training=False)
    output_columns = ["object_id"] + ["class_{label}".format(label=label) for label in one_hot_lookup.values()]
    first_insert = True
    num_objects_written = 0
    output = None

    for batch in stream_test_set():
        data = pd.DataFrame(batch)
        data = utils.pre_process(data, meta_data)

        if isinstance(output, pd.DataFrame):
            assert ~data.index.isin(output.object_id).all()

        data[feature_columns] = (data[feature_columns] - mean) / variance
        sorted_batch_ids, seq_tensor, seq_lengths = batch_generator(
            data=data,
            batch_ids=data.index.unique()
        )
        predicted_classes = run_inference(
            seq_tensor=seq_tensor,
            sequence_lengths=seq_lengths
        )
        inference_output = np.hstack((sorted_batch_ids[:, None], predicted_classes))
        output = pd.DataFrame(inference_output, columns=output_columns).sort_values("object_id")
        output.object_id = output.object_id.astype(int)

        if first_insert:
            output.to_csv(root + "/output/submission.csv", index=False)
            first_insert = False
        else:
            output.to_csv(root + "/output/submission.csv", mode="a", header=False, index=False)

        num_objects_written += len(output)
        print(num_objects_written, "written to submission output")


def deduplicate():
    data = pd.read_csv(root + "/output/submission.csv")
    data = data.drop_duplicates(subset="object_id", keep="first")
    data.to_csv(root + "/output/submission_dedupe.csv", index=False)
    return


if __name__ == '__main__':
    run()
    # deduplicate()
