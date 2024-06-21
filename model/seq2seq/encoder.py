import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, source_dict_size, embed_dim, lstm_hidden, lstm_layers, lstm_dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(source_dict_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim,
                            lstm_hidden,
                            lstm_layers,
                            dropout=lstm_dropout,
                            batch_first=True)

    def forward(self, source_sentence):
        embedded = self.embedding(source_sentence)

        outputs, (hidden, cell) = self.lstm(embedded)

        return outputs, (hidden, cell)
