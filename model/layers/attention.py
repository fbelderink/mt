import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention layer, implemented as proposed in Bahdanau & Cho
    """


    def __init__(self, encoder_hidden, decoder_hidden, use_dot_product = False   ):
        super(Attention, self).__init__()
        self.enc_dim = encoder_hidden
        self.use_dot_product = use_dot_product
        if self.use_dot_product:
            #Query Matrix for the "Query" from our decoder
            #Keys Matrix for encoder outputs
            # Values matrix for the actual context of our encoder
            self.Q = nn.Linear(decoder_hidden, encoder_hidden)
            #choice of our dimensions might be abritrarily
            self.K = nn.Linear(encoder_hidden ,encoder_hidden)

            self.V = nn.Linear(encoder_hidden, encoder_hidden)
        else:

            self.Wa = nn.Linear(decoder_hidden, encoder_hidden)
            self.Ua = nn.Linear(encoder_hidden, encoder_hidden)
            self.activation = nn.Tanh()

            self.Va = nn.Linear(encoder_hidden, 1)

    #implement scaled dot product
    def forward(self, encoder_outputs, decoder_hidden):
        if self.use_dot_product:
            #use pytorch sdpa
            #didnt specify scale here
            weights = torch.nn.functional.scaled_dot_product_attention(decoder_hidden, encoder_outputs, encoder_outputs)
            context = torch.bmm(weights, encoder_outputs)
            return context


            '''query = self.Q(decoder_hidden)
            keys = self.K(encoder_outputs)
            values = self.V(encoder_outputs)
            #QK^T
            qk_product = torch.bmm(query, keys.transpose(1, 2))
            #norm with scalar
            qk_product = qk_product * 1/(self.enc_dim ** 0.5)
            #scores = scores.permute(0,2,1)
            weights = F.softmax(qk_product, dim=1)

            context_vector = torch.bmm(weights, values)
            return context_vector'''


        else:


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

            # TODO mask padding

            context = torch.bmm(weights, encoder_outputs)  # (B x 1 x encoder_hidden)

            return context
