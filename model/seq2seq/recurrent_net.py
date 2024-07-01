import torch.nn as nn
import torch

from model.basic_net import BasicNet
from model.seq2seq.encoder import Encoder
from model.seq2seq.decoder import AttentionDecoder
from utils.model_hyperparameters import RNNModelHyperparameters
from preprocessing.dictionary import PADDING
from utils.types import RNNType


class RecurrentNet(BasicNet):
    def __init__(self,
                 source_dict_size,
                 target_dict_size,
                 config: RNNModelHyperparameters,
                 model_name="rnn"):
        super(RecurrentNet, self).__init__(source_dict_size, target_dict_size, config, model_name)

        if config.rnn_type == 'lstm':
            rnn_type = RNNType.LSTM
        elif config.rnn_type == 'gru':
            rnn_type = RNNType.GRU
        else:
            raise ValueError("Unsupported rnn type")

        self.encoder = Encoder(rnn_type,
                               source_dict_size,
                               config.encoder_parameters[0],
                               config.rnn_hidden_dim,
                               config.rnn_layers,
                               config.encoder_parameters[2],
                               config.encoder_parameters[1])

        self.decoder = AttentionDecoder(rnn_type,
                                        target_dict_size,
                                        config.decoder_parameters[0],
                                        config.rnn_hidden_dim,
                                        config.rnn_layers,
                                        config.decoder_parameters[1],
                                        config.use_attention,
                                        use_attention_dp=config.use_attention_dp,
                                        bidirectional_encoder=config.encoder_parameters[1])

        self.criterion = nn.CrossEntropyLoss(ignore_index=PADDING)

    def forward(self, source, target,
                teacher_forcing=False, apply_log_softmax=True):
        encoder_outputs, encoder_state = self.encoder(source)

        # encoder_outputs shape: [B x seq_len x directions*hidden]
        # encoder_state shapes: [directions*layers x B x hidden]
        # (includes cell state and hidden state)
        # i.e. the two directions are already concatenated

        decoder_outputs = self.decoder(encoder_outputs, encoder_state, target,
                                       teacher_forcing=teacher_forcing,
                                       apply_log_softmax=apply_log_softmax)

        decoder_outputs = decoder_outputs.permute(0, 2, 1)

        return decoder_outputs

    def compute_loss(self, pred, label):
        # pred shape (B x target_dict_size x seq_len)

        return self.criterion(pred, label)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
