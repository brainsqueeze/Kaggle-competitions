from lsst_project.src.model import Classifier
from lsst_project.data import utils

import pandas as pd
import torch


def log(message):
    m = "[INFO] {message}".format(message=message)
    print(m)


def pre_process(data, meta_data):
    assert isinstance(data, pd.DataFrame)
    data["unix_time"] = utils.mjd_to_unix_time(data.mjd)
    data = data.join(meta_data, on="object_id", lsuffix="_main", rsuffix="_meta")

    data["distmod"].fillna(0, inplace=True)
    return data


def batch_generator(data, batch_ids, max_length, columns, one_hot_lookup):
    data = data[data.object_id_main.isin(batch_ids)]
    lengths = data.groupby(by="object_id_main").size().as_matrix()
    x_train = [data[data.object_id_main == object_id][columns].as_matrix() for object_id in batch_ids]

    target = data[data.object_id_main.isin(batch_ids)].groupby("object_id_main").target.unique().astype(float).values
    target = utils.encode_targets(target_array=target, col_lookup=one_hot_lookup)

    seq_tensor = torch.autograd.Variable(torch.zeros((len(x_train), max_length, len(columns)))).float()
    for idx, (seq, seqlen) in enumerate(zip(x_train, lengths)):
        seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.float32)
    return seq_tensor, torch.tensor(target, dtype=torch.long), lengths


def run():
    log("Loading data")
    meta = utils.load_meta_data(training=True)
    data = utils.load_data(training=True)
    data = pre_process(data=data, meta_data=meta)

    log("Getting longest sequence for padding information")
    lengths = data.groupby(by="object_id_main").size().as_matrix()

    columns_to_exclude = {
        "object_id_main",
        "object_id_meta",
        "mjd",
        "unix_time",
        "detected",
        "hostgal_photoz",
        "target"
    }
    feature_columns = [col for col in data.columns if col not in columns_to_exclude]

    max_length = lengths.max()
    classes = sorted(meta.target.unique()) + [99]
    num_targets = len(classes)
    one_hot_lookup = {v: idx for idx, v in enumerate(classes)}

    loss = torch.nn.CrossEntropyLoss()
    model = Classifier(
        num_features=len(feature_columns),
        num_classes=num_targets,
        max_sequence_length=max_length
    )

    # todo: replace the code below with actual mini-batching code, this is just unit testing
    ids = [615, 713, 730]
    seq_tensor, target, train_lengths = batch_generator(
        data=data,
        batch_ids=ids,
        max_length=max_length,
        columns=feature_columns,
        one_hot_lookup=one_hot_lookup
    )

    output = model.forward(x=seq_tensor, sequence_lengths=train_lengths)
    cost = loss(output, target)
    return cost


if __name__ == '__main__':
    run()
