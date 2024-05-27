import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__()

        self.has_bias = bias
        self.W = nn.Parameter(torch.empty(in_features, out_features))

        nn.init.xavier_normal_(self.W)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        output = torch.matmul(x, self.W)

        if self.has_bias:
            output += self.bias

        return output




