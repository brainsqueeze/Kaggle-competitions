import tensorflow as tf
import numpy as np
from quora_project.src.models import CnnLstm

from quora_project.src.dictionary import EmbeddingLookup
from quora_project.src import training_utils as utils

from quora_project import config

import pickle
import json
import os


def train(model_folder, num_tokens=10000, num_hidden=128, conv_size=128,
          batch_size=32, num_batches=50, num_epochs=10,
          use_tf_idf=False):

    log_dir = config.MODEL_PATH + model_folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    utils.log("Fetching corpus and transforming to frequency domain")
    corpus = utils.load_text()

    utils.log("Sub-sampling to balance classes")
    corpus = utils.sub_sample(data=corpus, split_type="skew", skew=3)

    utils.log("Splitting the training and validation sets")
    train_data, cv_data = utils.test_val_split(corpus=corpus, val_size=512)

    t_ids, t_texts, t_targets = train_data
    cv_ids, cv_texts, cv_targets = cv_data

    utils.log("Fitting embedding lookup and transforming the training and cross-validation sets")
    lookup = EmbeddingLookup(top_n=num_tokens, use_tf_idf_importance=use_tf_idf)
    full_text = lookup.fit_transform(corpus=t_texts)
    cv_x = lookup.transform(corpus=cv_texts)

    utils.log("Removing empty sequences")
    t_ids, full_text, t_targets = utils.remove_null_sequences(ids=t_ids, sequences=full_text, targets=t_targets)
    cv_ids, cv_x, cv_targets = utils.remove_null_sequences(ids=cv_ids, sequences=cv_x, targets=cv_targets)

    utils.log("Getting the maximum sequence length and vocab size")
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

        utils.log(f"Padding sequences in corpus to length {max_seq_len}")
    full_text = np.array([utils.pad_sequence(seq, max_seq_len) for seq in full_text])
    cv_x = np.array([utils.pad_sequence(seq, max_seq_len) for seq in cv_x])
    keep_probabilities = [0.5, 0.6, 1.0]

    utils.log("Compiling seq2seq automorphism model")
    seq_input = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len])
    target_input = tf.placeholder(dtype=tf.float32, shape=[None, ])
    keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

    file_sys = open(log_dir + "/model.json", "w")
    meta_data = {
        "embeddingDimensions": num_hidden,
        "maxSequenceLength": max_seq_len,
        "vocabSize": vocab_size,
        "convWeightDim": conv_size,
        "trainingParameters": {
            "keepProbabilities": keep_probabilities,
            "nBatches": num_batches,
            "batchSize": batch_size,
            "maxEpochs": num_epochs
        }
    }
    json.dump(meta_data, file_sys, indent=2)
    file_sys.close()

    model = CnnLstm(
        input_x=seq_input,
        embedding_size=300,
        vocab_size=vocab_size,
        keep_prob=keep_prob,
        num_hidden=num_hidden,
        conv_size=conv_size,
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

        model.assign_lr(sess, 0.1)
        model.assign_clip_norm(sess, 10.0)

        for epoch in range(num_epochs):
            print("\t Epoch: {0}".format(epoch + 1))
            train_summary = tf.Summary()
            i = 1
            train_predictions, train_labels = [], []

            for x, y in utils.mini_batches(
                    (full_text, t_targets),
                    size=batch_size,
                    n_batches=num_batches,
                    max_len=max_seq_len,
                    seed=epoch
            ):
                if x.shape[0] == 0:
                    continue

                loss_val, gradient, predictions, _ = sess.run(
                    [model.loss, model.gradient_norm, model.predict, model.train],
                    feed_dict={seq_input: x, target_input: y, keep_prob: keep_probabilities}
                )
                train_predictions.append(predictions)
                train_labels.append(y)

                train_summary.value.add(tag="cost", simple_value=loss_val)
                train_summary.value.add(tag="gradient_norm", simple_value=gradient)
                summary_writer_train.add_summary(train_summary, epoch * num_batches + i)

                if i % (num_batches // 10) == 0:
                    print("\t\t iteration {0} - loss: {1}".format(i, loss_val))
                i += 1

            train_predictions = np.concatenate(train_predictions)
            train_labels = np.concatenate(train_labels)
            f1_score = utils.f1(predicted=train_predictions, expected=train_labels)
            train_summary.value.add(tag="f1-score", simple_value=f1_score)
            summary_writer_train.add_summary(train_summary, epoch * num_batches + i)
            summary_writer_train.flush()

            dev_summary = tf.Summary()
            cv_loss, predictions = sess.run(
                [model.loss, model.predict],
                feed_dict={seq_input: cv_x, target_input: cv_targets, keep_prob: keep_probabilities}
            )
            f1_score = utils.f1(predicted=predictions, expected=cv_targets)
            utils.log(f"Cross-validation F1-score: {f1_score}")
            dev_summary.value.add(tag="cost", simple_value=cv_loss)
            dev_summary.value.add(tag="f1-score", simple_value=f1_score)
            summary_writer_dev.add_summary(dev_summary, epoch * num_batches + i)
            summary_writer_dev.flush()

            lstm_file_name = saver.save(sess, log_dir + '/embedding_model', global_step=int((epoch + 1) * i))

    return lstm_file_name


if __name__ == '__main__':
    name = "quora_c_lstm_v1"

    train(
        model_folder=name,
        num_tokens=100000,
        num_hidden=128,
        conv_size=512,
        batch_size=64,
        num_batches=1000,
        num_epochs=100,
        use_tf_idf=False
    )
