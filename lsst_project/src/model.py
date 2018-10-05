import torch.nn.functional as func
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, num_features, num_classes, max_sequence_length,
                 lstm_dim=64, num_lstm_layers=1, lstm_dropout=0.):
        super(Classifier, self).__init__()
        conv_1_output_dim = lstm_dim // 2
        conv_2_output_dim = conv_1_output_dim // 2

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False
        )

        self.conv_1 = nn.Conv1d(
            in_channels=lstm_dim,
            out_channels=conv_1_output_dim,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(num_features=conv_1_output_dim)

        self.conv_2 = nn.Conv1d(
            in_channels=conv_1_output_dim,
            out_channels=conv_2_output_dim,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )

        # the minus - 2 is for dimensional losses during convolutional sweeps
        flat_dims = conv_2_output_dim * (max_sequence_length - 2)

        self.flat = flat_dims
        self.max_len = max_sequence_length
        self.conv1_dim = conv_1_output_dim
        self.conv2_dim = conv_2_output_dim

        self.dense = nn.Linear(in_features=flat_dims, out_features=num_classes)

    def forward(self, x, sequence_lengths, max_sequence_length):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths=sequence_lengths, batch_first=True)

        # ht is the hidden state for time-step = sequence length
        # ct is the cell state for time-step = sequence length
        output, (ht, ct) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=max_sequence_length, batch_first=True)

        output = output.transpose(1, 2)  # need to swap inputs and sequences for CNN layers
        x = self.conv_1(output)
        x = func.relu(self.batch_norm(x))
        x = func.relu(self.conv_2(x))

        # transpose back to the original shape
        x = x.transpose(1, 2)

        # flatten
        x = x.contiguous().view(x.shape[0], -1)
        return self.dense(x)
