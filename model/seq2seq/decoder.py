import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.attention import Attention


class Decoder(nn.Module):

    def __init__(self, target_dict_size,
                 embed_dim,
                 lstm_hidden,
                 lstm_layers,
                 lstm_dropout):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(target_dict_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden,
            lstm_layers,
            dropout=lstm_dropout,
            batch_first=True
        )

    def forward(self, target_word, state):
        embedded = self.embedding(target_word)

        decoder_output, state = self.lstm(embedded, state)

        return decoder_output, state


class AttentionDecoder(nn.Module):
    def __init__(self,
                 target_dict_size,
                 embed_dim,
                 lstm_hidden,
                 lstm_layers,
                 lstm_dropout):

        super(AttentionDecoder, self).__init__()

        self.decoder = Decoder(target_dict_size,
                               embed_dim,
                               lstm_hidden,
                               lstm_layers,
                               lstm_dropout)

        self.attention = Attention(lstm_hidden)

        self.fc = nn.Linear(2 * lstm_hidden, target_dict_size)

    def forward(self, encoder_outputs, encoder_state, target_tensor, T_max=-1,
                teacher_forcing=False, apply_log_softmax=True):

        prob_dists = []
        target_word = target_tensor[:, 0].unsqueeze(1)

        state = encoder_state

        T_max = max(T_max, target_tensor.shape[1])
        for k in range(1, T_max):
            fc_out, state = self.forward_step(encoder_outputs, state, target_word, apply_log_softmax)

            if teacher_forcing:
                target_word = target_tensor[:, k].unsqueeze(1)
            else:
                max_idx = torch.argmax(fc_out, dim=-1)
                target_word = max_idx.unsqueeze(1).detach()

            prob_dists.append(fc_out)

        return torch.stack(prob_dists).permute(1, 0, 2)

    def forward_step(self, encoder_outputs, state, target_word, apply_log_softmax=True):
        decoder_outputs, state = self.decoder(target_word, state)

        context_vector = self.attention(encoder_outputs, decoder_outputs)

        decoder_outputs = decoder_outputs.squeeze(1)

        fc_out = self.fc(torch.cat((context_vector, decoder_outputs), dim=-1))

        if not self.training and apply_log_softmax:
            fc_out = F.log_softmax(fc_out, dim=-1)

        return fc_out, state
