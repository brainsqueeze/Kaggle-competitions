import numpy as np
from quora_project.src import data_utils as utils

from quora_project import config

import csv


def log(message):
    print(f"[INFO] {message}")


def load_text(infer=False):
    path = config.DATA_PATH
    files = ["train.csv"]
    if infer:
        files = ["test.csv"]
    ids, texts, targets = [], [], []

    for file in files:
        with open(path + file, "r", encoding="latin1") as csv_f:
            reader = csv.DictReader(csv_f)
            for idx, row in enumerate(reader):
                # if idx > 1000:
                #     break
                ids.append(row["qid"])
                texts.append(utils.pre_process(row["question_text"]))
                targets.append(int(row["target"]))

    return np.array(ids), np.array(texts), np.array(targets, np.float32)


def sub_sample(data, split_type='balanced', skew=1):
    ids, texts, targets = data
    assert isinstance(targets, np.ndarray)

    size_pos_class = targets[targets == 1].shape[0]
    size_neg_class = targets[targets == 0].shape[0]
    assert split_type in {'balanced', 'skew'}
    if split_type == 'skew':
        assert isinstance(skew, int) and skew > 0

    if split_type == 'balanced':
        indices = np.random.permutation(range(size_neg_class))[:size_pos_class]
    elif split_type == 'skew':
        assert isinstance(skew, int) and skew > 0
        indices = np.random.permutation(range(size_neg_class))[:skew * size_pos_class]
    else:
        raise ValueError

    ids_ = np.concatenate((ids[targets == 1], ids[targets == 0][indices]))
    texts_ = np.concatenate((texts[targets == 1], texts[targets == 0][indices]))
    targets_ = np.concatenate((targets[targets == 1], targets[targets == 0][indices]))
    return ids_, texts_, targets_


def test_val_split(corpus, val_size):
    ids, texts, targets = corpus
    s = np.random.permutation(range(len(ids)))

    cv_ids = ids[s[:val_size]]
    cv_texts = texts[s[:val_size]]
    cv_targets = targets[s[:val_size]]

    train_ids = ids[s[val_size:]]
    train_texts = texts[s[val_size:]]
    train_targets = targets[s[val_size:]]
    return (train_ids, train_texts, train_targets), (cv_ids, cv_texts, cv_targets)


def remove_null_sequences(ids, sequences, targets):
    ids_, sequences_, targets_ = [], [], []
    for i in range(len(ids)):
        seq = sequences[i]
        if seq and len(seq) > 0:
            ids_.append(ids[i])
            sequences_.append(sequences[i])
            targets_.append(targets[i])
    return np.array(ids_), sequences_, np.array(targets_, dtype=np.float32)


def one_hot_encoded(targets):
    y = np.zeros(shape=(len(targets), 2), dtype=np.float32)
    y[targets == 0, 0] = 1
    y[targets == 1, 1] = 1
    return y, targets


def pad_sequence(sequence, max_sequence_length):
    """
    Pads individual text sequences to the maximum length
    seen by the model at training time
    :param sequence: list of integer lookup keys for the vocabulary (list)
    :param max_sequence_length: (int)
    :return: padded sequence (ndarray)
    """

    sequence = np.array(sequence, dtype=np.int32)
    difference = max_sequence_length - sequence.shape[0]
    pad = np.zeros((difference,), dtype=np.int32)
    return np.concatenate((sequence, pad))


def f1(predicted, expected):
    tp = np.sum(predicted * expected)
    fp = np.sum(predicted * (1 - expected))
    fn = np.sum((1 - predicted) * expected)

    precision = tp / (tp + fp) if tp + fp > 0 else 1
    recall = tp / (tp + fn) if tp + fn > 0 else 1

    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1_score


def mini_batches(corpus, size, n_batches, max_len, seed):
    np.random.seed(seed)
    sequences, targets = corpus
    s = np.random.choice(range(len(targets)), replace=False, size=min(len(targets), size * n_batches)).astype(np.int32)

    for mb in range(n_batches):
        mini_batch = s[mb * size: (mb + 1) * size]
        x = np.array([pad_sequence(sequences[index], max_len) for index in mini_batch])
        y = targets[mini_batch]
        yield x, y