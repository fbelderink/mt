import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.attention import Attention
from utils.types import RNNType
from preprocessing.dictionary import START, PADDING

"""
class Decoder(nn.Module):
    def __init__(self, target_dict_size, embed_dim):
        self.embedding = nn.Embedding(target_dict_size,
                                      embed_dim,
                                      padding_idx=PADDING)

        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def forward(self, encoder_output, hidden, target):



        for i in range(target.size(1) - 1):
            pass

    def step_forward(self, encoder_output, prev_state, target_tensor):
        pass

"""



class AttentionDecoder(nn.Module):
    def __init__(self,
                 rnn_type: RNNType,
                 target_dict_size,
                 embed_dim,
                 hidden,
                 layers,
                 dropout,
                 bidirectional,
                 use_attention=True):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(target_dict_size,
                                      embed_dim,
                                      padding_idx=PADDING)

        self.num_directions = 2 if bidirectional else 1
        self.layers = layers

        self.rnn_type = rnn_type

        if self.rnn_type == RNNType.GRU:
            self.rnn = nn.GRU(embed_dim + self.num_directions * hidden,
                              hidden,
                              layers,
                              dropout=dropout,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim + self.num_directions * hidden,
                               hidden,
                               layers,
                               dropout=dropout,
                               batch_first=True)

        self.use_attention = use_attention

        if self.use_attention:
            self.attention = Attention(self.num_directions * hidden, hidden)
            # attention output shape (B x 1 x encoder_hidden)

        self.fc = nn.Linear(hidden, target_dict_size)

    # add the encoder state of each directions together in order to fit them into the 
    # uni-directional encoder
    def sum_directions(self, encoder_state_entry):
        _, B, hidden = encoder_state_entry.shape
        reshaped = encoder_state_entry.view(self.num_directions, self.layers, B, hidden)
        summed = reshaped.sum(dim=0)

        return summed

    def forward(self, encoder_outputs, encoder_state, target_tensor,
                teacher_forcing=False, apply_log_softmax=True):

        prob_dists = []
        # expected target_tensor shape: (B x seq_len)
        target_word = target_tensor[:, 1].unsqueeze(1)


        # prev_decoder_state has to have dimensions
        #([directions * layers x B x hidden], [directions*layers x B x hidden])
        if self.num_directions == 1:
            prev_decoder_state = encoder_state
        else:
            prev_decoder_state = (self.sum_directions(encoder_state[0]),self.sum_directions(encoder_state[1]))

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

        if self.use_attention:
            if self.rnn_type == RNNType.GRU:
                prev_hidden_state = prev_state
            else:
                (prev_hidden_state, _) = prev_state

            last_hidden_state = prev_hidden_state.permute(1, 0, 2)[:, -1, :].unsqueeze(1)
            # last_hidden_state shape (B x 1 x lstm_hidden)
            context_vector = self.attention(encoder_outputs, last_hidden_state)

            # context_vector shape (B x 1 x hidden*directions)
        else:
            context_vector = encoder_outputs[:, -1, :].unsqueeze(1)

        concat = torch.cat((embedded, context_vector), dim=-1)
        # concat shape (B x 1 x (embed_dim + encoder_hidden))
        decoder_outputs, prev_state = self.rnn(concat, prev_state)
        # decoder_outputs shape (B x 1 x lstm_hidden * num_directions)

        fc_out = self.fc(decoder_outputs)

        if not self.training and apply_log_softmax:
            fc_out = F.log_softmax(fc_out, dim=-1)

        return fc_out, prev_state
