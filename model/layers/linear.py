import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, batch_size: int, in_features: int, out_features: int, bias=True):
        super().__init__()

        self.has_bias = bias

        self.W = nn.Parameter(torch.rand(batch_size, out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(batch_size, out_features))

    def forward(self, S, T):
        x = torch.cat((S, T), dim=1)
        if self.has_bias:
            return torch.add(torch.bmm(self.W, x), self.bias)

        return torch.bmm(self.W, x)
