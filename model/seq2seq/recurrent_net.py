import torch.nn as nn
from model.basic_net import BasicNet
from model.seq2seq.encoder import Encoder
from model.seq2seq.decoder import AttentionDecoder
from utils.hyperparameters.model_hyperparameters import RNNModelHyperparameters


class RecurrentNet(BasicNet):
    def __init__(self, source_dict_size, target_dict_size, config: RNNModelHyperparameters, eos_idx, model_name="rnn"):
        super(RecurrentNet, self).__init__(source_dict_size, target_dict_size, config, model_name)

        self.encoder = Encoder(source_dict_size,
                               config.encoder_parameters[0],
                               config.lstm_hidden_dim,
                               config.lstm_layers,
                               config.encoder_parameters[1],
                               config.lstm_bidirectional)

        self.decoder = AttentionDecoder(target_dict_size,
                                        config.decoder_parameters[0],
                                        config.lstm_hidden_dim,
                                        config.lstm_layers,
                                        config.decoder_parameters[1],
                                        config.lstm_bidirectional)

        self.criterion = nn.CrossEntropyLoss(ignore_index=eos_idx)

    def forward(self, S, T, T_max=-1,
                teacher_forcing=False, apply_log_softmax=True):

        encoder_outputs, encoder_state = self.encoder(S)
        decoder_outputs = self.decoder(encoder_outputs, encoder_state, T,
                                       teacher_forcing=teacher_forcing,
                                       T_max=T_max,
                                       apply_log_softmax=apply_log_softmax)

        return decoder_outputs

    def compute_loss(self, pred, L):
        pred = pred.permute(0, 2, 1)

        return self.criterion(pred, L)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
