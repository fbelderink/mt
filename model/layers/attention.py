import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention layer, implemented as proposed in Bahdanau & Cho
    """

    def __init__(self, encoder_hidden, decoder_hidden):
        super(Attention, self).__init__()

        self.Wa = nn.Linear(decoder_hidden, encoder_hidden)
        self.Ua = nn.Linear(encoder_hidden, encoder_hidden)
        self.activation = nn.Tanh()

        self.Va = nn.Linear(encoder_hidden, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs shape (B x L x encoder_hidden)
        # decoder_hidden shape (B x 1 x decoder_hidden)
        Wa = self.Wa(decoder_hidden)
        # expected shape (B x 1 x encoder_hidden)
        Ua = self.Ua(encoder_outputs)
        # expected shape (B x L x encoder_hidden)

        scores = self.Va(self.activation(Wa + Ua))
        scores = scores.permute(0, 2, 1)
        # expected shape (B x 1 x seq_len)

        weights = F.softmax(scores, dim=-1)
        # expected shape (B x 1 x seq_len)

        context = torch.bmm(weights, encoder_outputs)  # (B x 1 x encoder_hidden)

        return context
