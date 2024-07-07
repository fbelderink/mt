import torch.nn as nn

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
                                        use_attention_mask=config.use_attention_mask,
                                        attn_dropout=config.attention_dropout,
                                        bidirectional_encoder=config.encoder_parameters[1],
                                        hidden_ll=config.hidden_ll,
                                        num_ll=config.num_ll,
                                        dropout_ll=config.dropout_ll,
                                        batch_norm_ll=config.batch_norm_ll,
                                        activation_function_ll=config.activation_function_ll)

        self.criterion = nn.CrossEntropyLoss(ignore_index=PADDING)

    def forward(self, source, target,
                teacher_forcing_ratio=0,
                apply_log_softmax=True):
        # source shape: (B x seq_len)
        # target shape: (B x 1)

        encoder_outputs, encoder_state = self.encoder(source)

        attn_mask = source != PADDING
        attn_mask = attn_mask.unsqueeze(1)

        # encoder_outputs shape: [B x seq_len x directions*hidden]
        # encoder_state shapes: [directions*layers x B x hidden]
        # (includes cell state and hidden state)
        # i.e. the two directions are already concatenated

        decoder_outputs = self.decoder(encoder_outputs, encoder_state, target,
                                       teacher_forcing_ratio=teacher_forcing_ratio,
                                       apply_log_softmax=apply_log_softmax,
                                       attn_mask=attn_mask)

        decoder_outputs = decoder_outputs.permute(0, 2, 1)

        return decoder_outputs

    def compute_loss(self, pred, label):
        # pred shape (B x target_dict_size x seq_len)

        return self.criterion(pred, label)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
