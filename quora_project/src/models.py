import tensorflow as tf


def build_cell(num_layers, num_hidden, keep_prob, use_cuda=False):
    cells = []

    for _ in range(num_layers):
        if use_cuda:
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden)
        else:
            cell = tf.nn.rnn_cell.LSTMCell(
                num_hidden,
                forget_bias=0.0,
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
            )

        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        cells.append(cell)

    if num_layers > 1:
        return tf.nn.rnn_cell.MultiRNNCell(cells)
    return cells[0]


def concat_reducer(seq_fw, seq_bw):
    if tf.contrib.framework.nest.is_sequence(seq_fw):
        tf.contrib.framework.nest.assert_same_structure(seq_fw, seq_bw)

        x_flat = tf.contrib.framework.nest.flatten(seq_fw)
        y_flat = tf.contrib.framework.nest.flatten(seq_bw)

        flat = []
        for x_i, y_i in zip(x_flat, y_flat):
            flat.append(tf.concat([x_i, y_i], axis=-1))

        return tf.contrib.framework.nest.pack_sequence_as(seq_fw, flat)
    return tf.concat([seq_fw, seq_bw], axis=-1)


class AttentionSVM(object):

    def __init__(self, input_x, vocab_size, embedding_size, keep_prob, num_hidden, attention_size,
                 is_training=False, input_y=None):
        self._batch_size, self._time_steps = input_x.get_shape().as_list()
        self._num_hidden = num_hidden
        self._attention_size = attention_size

        self._use_cuda = tf.test.is_gpu_available()

        self._input_keep_prob, self._lstm_keep_prob, self._dense_keep_prob = tf.unstack(keep_prob)

        # input embedding
        with tf.variable_scope('embedding'):
            self._seq_lengths = tf.count_nonzero(input_x, axis=1, name="sequence_lengths")
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="word_embeddings",
                dtype=tf.float32,
                trainable=True
            )
            self._input = tf.nn.embedding_lookup(embeddings, input_x)
            input_x = self.__input_op()

        # bi-directional encoder
        self.__context, final_state, shape = self.__encoder(input_x)

        # dense layer
        self.__output = self.__dense(self.__context)
        self.__predictions = tf.sign(tf.maximum(self.__output, 0))

        if is_training:
            assert input_y is not None
            self.target = input_y * 2 - 1
            self._input = tf.stack(self._input)

            with tf.variable_scope('cost'):
                self.loss = self.__cost()

            with tf.variable_scope('optimizer'):
                self._lr = tf.Variable(0.0, trainable=False)
                self._clip_norm = tf.Variable(0.0, trainable=False)
                t_vars = tf.trainable_variables()

                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), self._clip_norm)
                opt = tf.train.GradientDescentOptimizer(self._lr)

                # compute the gradient norm - only for logging purposes - remove if greatly affecting performance
                self.gradient_norm = tf.sqrt(sum([tf.norm(t) ** 2 for t in grads]), name="gradient_norm")

                self.train = opt.apply_gradients(
                    zip(grads, t_vars),
                    global_step=tf.train.get_or_create_global_step()
                )

                self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
                self._lr_update = tf.assign(self._lr, self._new_lr)

                self._new_clip_norm = tf.placeholder(tf.float32, shape=[], name="new_clip_norm")
                self._clip_norm_update = tf.assign(self._clip_norm, self._new_clip_norm)

    def assign_lr(self, session, lr_value):
        """
        Updates the learning rate
        :param session: (TensorFlow Session)
        :param lr_value: (float)
        """

        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_clip_norm(self, session, norm_value):
        """
        Updates the gradient normalization factor
        :param session: (TensorFlow Session)
        :param norm_value: (float)
        """

        session.run(self._clip_norm_update, feed_dict={self._new_clip_norm: norm_value})

    def __input_op(self):
        with tf.variable_scope('input_dropout'):
            input_x = self._input
            return tf.nn.dropout(x=input_x, keep_prob=self._input_keep_prob)

    def __encoder(self, input_x):
        with tf.variable_scope('encoder'):
            num_layers = 2

            forward = build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self._use_cuda
            )
            backward = build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self._use_cuda
            )

            output_seq, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward,
                cell_bw=backward,
                inputs=input_x,
                dtype=tf.float32,
                sequence_length=self._seq_lengths
            )

            # perform element-wise summations to combine the forward and backward sequences
            encoder_output = concat_reducer(seq_fw=output_seq[0], seq_bw=output_seq[1])
            encoder_state = final_state

            # get the context vectors from the attention mechanism
            context = self.__attention(input_x=encoder_output)

            return context, encoder_state, tf.shape(encoder_output)

    def __attention(self, input_x):
        with tf.variable_scope('source_attention'):
            in_dim = input_x.get_shape().as_list()[-1]
            w_omega = tf.get_variable(
                "w_omega",
                shape=[in_dim, self._attention_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            b_omega = tf.get_variable("b_omega", shape=[self._attention_size], initializer=tf.zeros_initializer())
            u_omega = tf.get_variable("u_omega", shape=[self._attention_size], initializer=tf.zeros_initializer())

            v = tf.tanh(tf.einsum("ijk,kl->ijl", input_x, w_omega) + b_omega)
            vu = tf.einsum("ijl,l->ij", v, u_omega, name="Bahdanau_score")
            alphas = tf.nn.softmax(vu, name="attention_weights")

            output = tf.reduce_sum(input_x * tf.expand_dims(alphas, -1), 1, name="context_vector")
            return output

    def __dense(self, input_x):
        with tf.variable_scope('dense'):
            input_dim = input_x.get_shape().as_list()[-1]
            input_x = tf.nn.dropout(input_x, keep_prob=self._dense_keep_prob, name="dense_dropout")

            weight = tf.get_variable(
                "dense_weight",
                shape=[input_dim],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(-0.01, 0.01)
            )

            bias = tf.get_variable(
                "dec_bias",
                shape=[],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            output = tf.tensordot(input_x, weight, axes=1) - bias
            return output

    def __cost(self):
        with tf.variable_scope('linear_svm_loss'):
            projection_dist = 1 - self.target * self.__output
            margin = tf.nn.relu(projection_dist)
            loss = tf.reduce_mean(margin ** 2)

        with tf.variable_scope('l2_loss'):
            weights = tf.trainable_variables()

            # only perform L2-regularization on the fully connected layer(s)
            l2_losses = [tf.nn.l2_loss(v) for v in weights if 'dense_weight' in v.name]
            loss += 1e-3 * tf.add_n(l2_losses)
        return loss

    @property
    def lr(self):
        return self._lr

    @property
    def clip_norm(self):
        return self._clip_norm

    @property
    def margin_distance(self):
        return self.__output

    @property
    def predict(self):
        return self.__predictions
