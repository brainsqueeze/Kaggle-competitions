from lsst_project.src.model import Classifier
from lsst_project.data import utils

import pandas as pd
import numpy as np
import torch
from torch.nn import functional as f

from lsst_project.data import config


def log(message):
    m = "[INFO] {message}".format(message=message)
    print(m)


def pre_process(data, meta_data):
    assert isinstance(data, pd.DataFrame)
    data["unix_time"] = utils.mjd_to_unix_time(data.mjd)
    data = data.set_index('object_id').join(meta_data.set_index('object_id'))

    data["distmod"].fillna(0, inplace=True)
    return data


def scale_data(data, columns):
    features = data[columns]
    mean, variance = features.mean(), features.std()

    # m_dict, v_dict = mean.to_dict(), variance.to_dict()
    # todo store this lookup for retrieval at inference time
    # lookup = {col: {"mean": avg, "variance": var} for (col, avg), (_, var) in zip(m_dict.items(), v_dict.items())}

    data[columns] -= mean
    data[columns] /= variance
    return data, (mean, variance)


def shuffle_sample(object_ids, batch_size, num_batches, seed=0):
    np.random.seed(seed)
    ids = np.random.permutation(object_ids)

    for batch_num in range(num_batches):
        yield ids[batch_num * batch_size: (batch_num + 1) * batch_size]


def batch_generator(data, batch_ids, max_length, columns, one_hot_lookup):
    data = data[data.index.isin(batch_ids)]

    # need to sort everything in descending by sequence lengths
    # because that is the format that sequence packing method expects
    lengths = data.groupby(data.index).size()[batch_ids].sort_values(ascending=False)
    sorted_batch_ids = lengths.index
    x_train = [data[data.index == object_id][columns].values for object_id in sorted_batch_ids]

    target = data.groupby(data.index).target.unique()[sorted_batch_ids].astype(float).values
    target = utils.encode_targets(target_array=target, col_lookup=one_hot_lookup)

    seq_tensor = torch.autograd.Variable(torch.zeros((len(x_train), max_length, len(columns)))).float()
    for idx, (seq, seqlen) in enumerate(zip(x_train, lengths)):
        seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.float32)
    return seq_tensor, torch.tensor(target, dtype=torch.long), lengths.values


def compute_f1(predictions, expectations, num_classes):
    y = np.zeros((len(expectations), num_classes), dtype=np.float32)
    y_hat = np.zeros((len(predictions), num_classes), dtype=np.float32)

    y[np.arange(y.shape[0]), expectations] = 1
    y_hat[np.arange(y_hat.shape[0]), predictions] = 1

    with np.errstate(divide='ignore', invalid='ignore'):
        tp = np.sum(y_hat * y, axis=0)
        fp = np.sum(y_hat * (1 - y), axis=0)
        fn = np.sum((1 - y_hat) * y, axis=0)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precision[np.isnan(precision)] = 0
        recall[np.isnan(recall)] = 0

        f1 = 2 * precision * recall / (precision + recall)
        f1[np.isnan(f1)] = 0
        return f1, (precision, recall)


def run(num_epochs=2, batch_size=32, num_batches=50):
    log("Loading data")
    meta = utils.load_meta_data(training=True)
    data = utils.load_data(training=True)
    data = pre_process(data=data, meta_data=meta)

    log("Getting longest sequence for padding information")
    lengths = data.groupby(data.index).size().values

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

    log("Scaling data")
    data, _ = scale_data(data=data, columns=feature_columns)

    max_length = lengths.max()
    classes = sorted(meta.target.unique()) + [99]
    num_targets = len(classes)
    one_hot_lookup = {v: idx for idx, v in enumerate(classes)}
    class_lookup = {v: k for k, v in one_hot_lookup.items()}
    object_ids = meta.object_id.values

    model = Classifier(
        num_features=len(feature_columns),
        num_classes=num_targets,
        max_sequence_length=max_length
    )
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):

        log("\tEpoch: {epoch}".format(epoch=epoch + 1))
        running_loss = 0
        running_f1 = 0
        i = 0

        for batch_ids in shuffle_sample(object_ids, batch_size, num_batches, seed=0):
            seq_tensor, target, train_lengths = batch_generator(
                data=data,
                batch_ids=batch_ids,
                max_length=max_length,
                columns=feature_columns,
                one_hot_lookup=one_hot_lookup
            )

            if seq_tensor.shape[0] == 0:
                continue

            opt.zero_grad()
            outputs = model.forward(
                x=seq_tensor,
                sequence_lengths=train_lengths,
                max_sequence_length=max_length
            )
            cost = loss(outputs, target)
            cost.backward()
            opt.step()

            running_loss += cost.item()

            # get the predicted classes
            _, index = torch.max(f.softmax(outputs, dim=1), 1)
            f1, _ = compute_f1(predictions=index, expectations=target, num_classes=num_targets)
            running_f1 += f1.mean()

            if (i + 1) % 10 == 0:
                print("\t\tloss: {loss} | average F1: {f1}".format(loss=running_loss / 10, f1=running_f1 / 10))
                running_loss = 0.0
            i += 1

        torch.save(model.state_dict(), config.MODEL_PATH + "model.pth")


if __name__ == '__main__':
    run(num_epochs=10, batch_size=64, num_batches=50)
