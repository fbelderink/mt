import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.attention import Attention
from utils.types import RNNType
from preprocessing.dictionary import START, PADDING


class AttentionDecoder(nn.Module):
    def __init__(self,
                 rnn_type: RNNType,
                 target_dict_size,
                 embed_dim,
                 hidden,
                 layers,
                 dropout,
                 use_attention=True,
                 use_attention_dp=True,
                 bidirectional_encoder=False):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(target_dict_size,
                                      embed_dim,
                                      padding_idx=PADDING)

        self.layers = layers

        self.rnn_type = rnn_type

        self.bidirectional_encoder = bidirectional_encoder

        self.num_directions = 2 if self.bidirectional_encoder else 1

        if self.rnn_type == RNNType.GRU:
            self.rnn = nn.GRU(embed_dim,
                              self.num_directions * hidden,
                              layers,
                              dropout=dropout,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim,
                               self.num_directions * hidden,
                               layers,
                               dropout=dropout,
                               batch_first=True)

        self.use_attention = use_attention

        if self.use_attention:
            self.attention = Attention(self.num_directions * hidden,
                                       self.num_directions * hidden,
                                       use_dot_product=use_attention_dp)
            # attention output shape (B x 1 x encoder_hidden)

        self.fc = nn.Linear(2 * self.num_directions * hidden, target_dict_size)

    # add the encoder state of each direction together in order to fit them into the
    # uni-directional decoder
    def reshape_state(self, encoder_state_entry):
        _, B, hidden = encoder_state_entry.shape
        reshaped = encoder_state_entry.view(self.layers, B, hidden * self.num_directions)

        return reshaped

    def forward(self, encoder_outputs, encoder_state, target_tensor,
                teacher_forcing=False, apply_log_softmax=True):
        # encoder_outputs expected shape: (B x seq_len x hidden)
        # encoder_state expected shapes: ([directions * layers x B x hidden], [directions * layers x B x hidden])

        prob_dists = []
        # expected target_tensor shape: (B x seq_len)
        target_word = target_tensor[:, 1].unsqueeze(1)

        prev_decoder_state = (self.reshape_state(encoder_state[0]), self.reshape_state(encoder_state[1]))

        for k in range(1, target_tensor.size(1)):
            fc_out, prev_decoder_state = self.forward_step(encoder_outputs,
                                                           prev_decoder_state,
                                                           target_word,
                                                           apply_log_softmax)
            # fc_out shape (B x 1 x target_dict_size)

            if teacher_forcing:
                target_word = target_tensor[:, k].unsqueeze(1)
            else:
                target_word = torch.argmax(fc_out, dim=-1)

            prob_dists.append(fc_out)
        return torch.cat(prob_dists, dim=1)

    def forward_step(self,
                     encoder_outputs,
                     prev_state,
                     target_word,
                     apply_log_softmax=True):
        # encoder_outputs shape: (B x seq_len x hidden * directions)
        # hidden_state shape: (directions * num_layers x B x hidden)
        # target word shape: (B x 1)

        embedded = self.embedding(target_word)
        # embedded shape (B x 1 x embed_dim)
        decoder_outputs, prev_state = self.rnn(embedded, prev_state)
        # decoder_outputs shape (B x 1 x hidden * num_directions)

        if self.use_attention:
            context_vector = self.attention(encoder_outputs, decoder_outputs)
            # context_vector shape (B x 1 x encoder_hidden)
        else:
            context_vector = encoder_outputs[:, -1, :].unsqueeze(1) #TODO sinnfrei (müssen 1:1 alignment sonst machen eigentlich)

        concat = torch.cat((decoder_outputs, context_vector), dim=-1)
        # concat shape (B x 1 x (decoder_hidden + encoder_hidden))
        fc_out = self.fc(concat)

        if not self.training and apply_log_softmax:
            fc_out = F.log_softmax(fc_out, dim=-1)

        return fc_out, prev_state
