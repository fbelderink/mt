import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, batch_size: int, in_features: int, out_features: int, bias=True):
        super().__init__()

        self.has_bias = bias

        rand_weight_matrix = torch.rand(out_features, in_features) # init of rand weight matrix
        rand_weight_tensor = torch.stack([rand_weight_matrix] * batch_size, dim=0) # for processing batches simultaneously
        self.W = nn.Parameter(rand_weight_tensor) # added to parameters

        if bias:
            self.bias = nn.Parameter(torch.zeros(batch_size, out_features))

    def forward(self, S, T):
        x = torch.cat((S, T), dim=1)
        if self.has_bias:
            return torch.add(torch.bmm(self.W, x), self.bias)

        return torch.bmm(self.W, x)
