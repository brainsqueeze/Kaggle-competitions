import torch.nn.functional as f
import torch


class SvmClassifier(torch.nn.Module):

    def __init__(self, num_features, num_classes, lstm_dim=64, num_lstm_layers=1, attention_size=64,
                 lstm_dropout=0.0, dense_dropout=0.0):
        super(SvmClassifier, self).__init__()

        use_cuda = torch.cuda.is_available()

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False
        )

        self.w_attn = torch.randn(lstm_dim, attention_size, dtype=torch.float32)
        self.b_attn = torch.zeros(attention_size, dtype=torch.float32)
        self.u_attn = torch.zeros(attention_size, dtype=torch.float32)

        self.w_svm = torch.randn(lstm_dim, num_classes, dtype=torch.float32)
        self.b_svm = torch.zeros(num_classes, dtype=torch.float32)

        if use_cuda:
            self.w_attn = self.w_attn.cuda()
            self.b_attn = self.b_attn.cuda()
            self.u_attn = self.u_attn.cuda()

            self.w_svm = self.w_svm.cuda()
            self.b_svm = self.b_svm.cuda()

        self.probability_logits = torch.nn.Linear(in_features=num_classes, out_features=num_classes)

    def forward(self, x, sequence_lengths, max_sequence_length):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=sequence_lengths, batch_first=True)

        # ht is the hidden state for time-step = sequence length
        # ct is the cell state for time-step = sequence length
        output, (ht, ct) = self.lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, total_length=max_sequence_length, batch_first=True)

        context_vector = self.attention(output)
        svm_output = self.decision(context_vector)
        logits = self.probability_logits(svm_output)
        return svm_output, logits

    def attention(self, x):
        v = torch.einsum("ijk,kl->ijl", (x, self.w_attn)) + self.b_attn
        vu = torch.einsum("ijl,l->ij", (v, self.u_attn))
        alphas = f.softmax(vu, dim=-1)
        return torch.sum(x * alphas.unsqueeze(-1), dim=1)

    def decision(self, x):
        return torch.matmul(x, self.w_svm) - self.b_svm


class LstmCnnClassifier(torch.nn.Module):

    def __init__(self, num_features, num_classes, max_sequence_length, lstm_dim=64, num_lstm_layers=1,
                 lstm_dropout=0., dense_dropout=0., conv_dropout=0.):
        super(LstmCnnClassifier, self).__init__()
        max_sequence_length = int(max_sequence_length)
        conv_1_output_dim = lstm_dim // 2
        conv_2_output_dim = conv_1_output_dim // 2

        # the minus - 2 is for dimensional losses during convolutional sweeps
        flat_dims = conv_2_output_dim * (max_sequence_length - 2)

        self.flat = flat_dims
        self.max_len = max_sequence_length

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False
        )

        self.conv_1 = torch.nn.Conv1d(
            in_channels=lstm_dim,
            out_channels=conv_1_output_dim,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )

        self.batch_norm = torch.nn.BatchNorm1d(num_features=conv_1_output_dim)

        self.conv_2 = torch.nn.Conv1d(
            in_channels=conv_1_output_dim,
            out_channels=conv_2_output_dim,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv1_dim = conv_1_output_dim
        self.conv2_dim = conv_2_output_dim

        self.input_dropout = torch.nn.Dropout(p=dense_dropout)
        self.dense_dropout = torch.nn.Dropout(p=dense_dropout)
        self.conv_dropout = torch.nn.Dropout(p=conv_dropout)
        self.dense = torch.nn.Linear(in_features=flat_dims, out_features=num_classes)

    def forward(self, x, sequence_lengths, max_sequence_length):
        x = self.input_dropout(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=sequence_lengths, batch_first=True)

        # ht is the hidden state for time-step = sequence length
        # ct is the cell state for time-step = sequence length
        output, (ht, ct) = self.lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, total_length=max_sequence_length, batch_first=True)

        output = output.transpose(1, 2)  # need to swap inputs and sequences for CNN layers
        x = self.conv_1(output)
        x = self.batch_norm(x)
        x = f.relu(self.conv_dropout(x))
        x = f.relu(self.conv_2(x))

        # transpose back to the original shape
        x = x.transpose(1, 2)

        # flatten
        x = x.reshape(x.shape[0], -1)
        x = self.dense_dropout(x)
        return self.dense(x)
