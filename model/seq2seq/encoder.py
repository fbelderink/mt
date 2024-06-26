import torch.nn as nn
import torch.nn.functional as F
from preprocessing.dictionary import PADDING
from utils.types import RNNType


class Encoder(nn.Module):
    def __init__(self,
                 rnn_type: RNNType,
                 source_dict_size,
                 embed_dim,
                 hidden,
                 layers,
                 dropout,
                 bidirectional):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(source_dict_size,
                                      embed_dim,
                                      padding_idx=PADDING)

        if rnn_type == RNNType.GRU:
            self.rnn = nn.GRU(embed_dim,
                              hidden,
                              layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=True)
        elif rnn_type == RNNType.LSTM:
            self.rnn = nn.LSTM(embed_dim,
                               hidden,
                               layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               batch_first=True)

    def forward(self, source_sentence):
        embedded = self.embedding(source_sentence)


        #embedded = F.relu(embedded) # TODO

        outputs, encoder_state = self.rnn(embedded)

        return outputs, encoder_state
