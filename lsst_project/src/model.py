import torch.nn.functional as func
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, num_features, num_classes,
                 lstm_dim=64, num_lstm_layers=1, lstm_dropout=0.):
        super(Classifier, self).__init__()
        self.__lstm_dims = lstm_dim
        self.__lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False
        )

        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x, sequence_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths=sequence_lengths, batch_first=True)

        # ht is the hidden state for time-step = sequence length
        # ct is the cell state for time-step = sequence length
        output, (ht, ct) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        x = func.relu(self.conv1(x))
        return func.relu(self.conv2(x))
