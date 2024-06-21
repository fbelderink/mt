import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention layer, implemented as proposed in Bahdanau & Cho
    """

    def __init__(self, hidden, bidirectional_lstms):
        super(Attention, self).__init__()

        if bidirectional_lstms:
            input_size = 2 * 2 * hidden
        else:
            input_size = 2 * hidden  # encoder_hidden + decoder_hidden

        self.fc_1 = nn.Linear(input_size, hidden)
        self.activation = nn.Tanh()

        self.v_a = nn.Linear(hidden, 1)

    def forward(self, encoder_outputs, decoder_outputs):
        # encoder_outputs shape (B x L x hidden)
        # decoder_outputs shape (B x 1 x hidden)
        decoder_outputs = decoder_outputs.repeat(1, encoder_outputs.size(1), 1)

        concat = torch.cat((encoder_outputs, decoder_outputs), dim=-1)

        out = self.fc_1(concat)
        out = self.activation(out)

        scores = self.v_a(out)

        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)

        return context



