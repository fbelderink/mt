import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention layer, implemented as proposed in Bahdanau & Cho
    """

    def __init__(self, encoder_hidden, decoder_hidden, use_dot_product=False, use_mask=False, attn_dropout=0.0):
        super(Attention, self).__init__()
        self.use_dot_product = use_dot_product
        self.use_mask = use_mask

        self.Wa = nn.Linear(decoder_hidden, encoder_hidden)
        self.Ua = nn.Linear(encoder_hidden, encoder_hidden)
        self.activation = nn.Tanh()

        self.Va = nn.Linear(encoder_hidden, 1)

        self.attn_dropout = attn_dropout

    def forward(self, encoder_outputs, decoder_outputs, attn_mask=None):
        # encoder_outputs expected shape: (B x seq_len x hidden)
        # decoder_outputs expected shape: (B x 1 x hidden)
        # attn_mask expected shape: (B x 1 x seq_len)

        if self.use_dot_product:
            # use pytorch sdpa
            # didnt specify scale here
            if self.use_mask:
                context = F.scaled_dot_product_attention(decoder_outputs,
                                                         encoder_outputs,
                                                         encoder_outputs,
                                                         attn_mask=attn_mask,
                                                         dropout_p=self.attn_dropout)
            else:
                context = F.scaled_dot_product_attention(decoder_outputs,
                                                         encoder_outputs,
                                                         encoder_outputs,
                                                         dropout_p=self.attn_dropout)

            return context
        else:
            Wa = self.Wa(decoder_outputs)
            # expected shape (B x 1 x encoder_hidden)
            Ua = self.Ua(encoder_outputs)
            # expected shape (B x L x encoder_hidden)

            scores = self.Va(self.activation(Wa + Ua))
            scores = scores.permute(0, 2, 1)
            # expected shape (B x 1 x seq_len)

            if self.use_mask:
                scores[attn_mask] = 0

            weights = F.softmax(scores, dim=-1)
            # expected shape (B x 1 x seq_len)
            # TODO dropout here

            context = torch.bmm(weights, encoder_outputs)  # (B x 1 x encoder_hidden)

            return context
