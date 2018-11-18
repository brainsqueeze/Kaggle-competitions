import tensorflow as tf
import numpy as np
from quora_project.src.models import AttentionSVM
from quora_project.src.dictionary import EmbeddingLookup

from quora_project import config

import pickle
import json
import csv
import os


def log(message):
    print(f"[INFO] {message}")


def load_text():
    path = config.DATA_PATH
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    ids, texts, targets = [], [], []

    for file in files:
        with open(path + file, "r", encoding="latin1") as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                ids.append(row["qid"])
                texts.append(row["question_text"])
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


def train(model_folder, num_tokens=10000, num_hidden=128, attention_size=128,
          batch_size=32, num_batches=50, num_epochs=10,
          use_tf_idf=False):

    log_dir = config.MODEL_PATH + model_folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log("Fetching corpus and transforming to frequency domain")
    corpus = load_text()

    log("Sub-sampling to balance classes")
    corpus = sub_sample(data=corpus, split_type="balanced")

    log("Splitting the training and validation sets")
    train_data, cv_data = test_val_split(corpus=corpus, val_size=512)

    t_ids, t_texts, t_targets = train_data
    cv_ids, cv_texts, cv_targets = cv_data

    log("Fitting embedding lookup and transforming the training and cross-validation sets")
    lookup = EmbeddingLookup(top_n=num_tokens, use_tf_idf_importance=use_tf_idf)
    full_text = lookup.fit_transform(corpus=t_texts)
    cv_x = lookup.transform(corpus=cv_texts)

    log("Removing empty sequences")
    t_ids, full_text, t_targets = remove_null_sequences(ids=t_ids, sequences=full_text, targets=t_targets)
    cv_ids, cv_x, cv_targets = remove_null_sequences(ids=cv_ids, sequences=cv_x, targets=cv_targets)

    log("Getting the maximum sequence length and vocab size")
    max_seq_len = max([len(seq) for seq in full_text + cv_x])
    vocab_size = max([max(seq) for seq in full_text + cv_x]) + 1

    with open(log_dir + "/lookup.pkl", "wb") as pf:
        pickle.dump(lookup, pf)

    # write word lookup to a TSV file for TensorBoard visualizations
    with open(log_dir + "/metadata.tsv", "w") as lf:
        reverse = lookup.reverse
        lf.write("<eos>\n")
        for k in reverse:
            lf.write(reverse[k] + '\n')

    log(f"Padding sequences in corpus to length {max_seq_len}")
    full_text = np.array([pad_sequence(seq, max_seq_len) for seq in full_text])
    train_y, t_targets = one_hot_encoded(targets=t_targets)
    cv_x = np.array([pad_sequence(seq, max_seq_len) for seq in cv_x])
    cv_y, cv_targets = one_hot_encoded(targets=cv_targets)
    keep_probabilities = [1.0, 0.7, 1.0]

    log("Compiling seq2seq automorphism model")
    seq_input = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len])
    target_input = tf.placeholder(dtype=tf.float32, shape=[None,])
    keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

    file_sys = open(log_dir + "/model.json", "w")
    meta_data = {
        "embeddingDimensions": num_hidden,
        "maxSequenceLength": max_seq_len,
        "vocabSize": vocab_size,
        "attentionWeightDim": attention_size,
        "trainingParameters": {
            "keepProbabilities": keep_probabilities,
            "nBatches": num_batches,
            "batchSize": batch_size,
            "maxEpochs": num_epochs
        }
    }
    json.dump(meta_data, file_sys, indent=2)
    file_sys.close()

    model = AttentionSVM(
        input_x=seq_input,
        embedding_size=512,
        vocab_size=vocab_size,
        keep_prob=keep_prob,
        num_hidden=num_hidden,
        attention_size=attention_size,
        is_training=True,
        input_y=target_input
    )

    lstm_file_name = None
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.8,
        allow_growth=True
    )
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False
    )

    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        summary_writer_train = tf.summary.FileWriter(log_dir + '/training', sess.graph)
        summary_writer_dev = tf.summary.FileWriter(log_dir + '/validation', sess.graph)
        summary_writer_train.add_graph(graph=sess.graph)
        summary_writer_train.flush()

        # add metadata to embeddings for visualization purposes
        config_ = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_conf = config_.embeddings.add()
        embeddings = sess.graph.get_tensor_by_name("embedding/word_embeddings:0")
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = log_dir + "/metadata.tsv"
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer_train, config_)

        model.assign_lr(sess, 0.01)
        model.assign_clip_norm(sess, 10.0)

        for epoch in range(num_epochs):
            print("\t Epoch: {0}".format(epoch + 1))
            train_summary = tf.Summary()
            i = 1

            for x, y in mini_batches(
                    (full_text, t_targets),
                    size=batch_size,
                    n_batches=num_batches,
                    max_len=max_seq_len,
                    seed=epoch
            ):
                if x.shape[0] == 0:
                    continue

                loss_val, gradient, _ = sess.run(
                    [model.loss, model.gradient_norm, model.train],
                    feed_dict={
                        seq_input: x,
                        target_input: y,
                        keep_prob: keep_probabilities
                    }
                )

                train_summary.value.add(tag="cost", simple_value=loss_val)
                train_summary.value.add(tag="gradient_norm", simple_value=gradient)
                summary_writer_train.add_summary(train_summary, epoch * num_batches + i)

                if i % (num_batches // 10) == 0:
                    print("\t\t iteration {0} - loss: {1}".format(i, loss_val))
                i += 1
            summary_writer_train.flush()

            dev_summary = tf.Summary()
            cv_loss, decision, predictions = sess.run(
                [model.loss, model.margin_distance, model.predict],
                feed_dict={seq_input: cv_x, target_input: cv_targets, keep_prob: keep_probabilities}
            )
            f1_score = f1(predicted=predictions, expected=cv_targets)
            dev_summary.value.add(tag="cost", simple_value=cv_loss)
            dev_summary.value.add(tag="f1-score", simple_value=f1_score)
            summary_writer_dev.add_summary(dev_summary, epoch * num_batches + i)
            summary_writer_dev.flush()

            lstm_file_name = saver.save(sess, log_dir + '/embedding_model', global_step=int((epoch + 1) * i))

    return lstm_file_name


if __name__ == '__main__':
    name = "quora_svm_v1"

    train(
        model_folder=name,
        num_tokens=20000,
        num_hidden=256,
        attention_size=128,
        batch_size=32,
        num_batches=50,
        num_epochs=100,
        use_tf_idf=False
    )
