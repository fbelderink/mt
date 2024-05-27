import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, batch_size: int, in_features: int, out_features: int, bias=False):
        super().__init__()

        self.has_bias = bias
        rand_weight_matrix = torch.rand(in_features, out_features)
        rand_weight_tensor = torch.stack([rand_weight_matrix] * batch_size, dim=0)
        self.W = nn.Parameter(rand_weight_tensor)

        if bias:
            self.bias = nn.Parameter(torch.zeros(batch_size, out_features))
        print("done initalizing layer")

    def forward(self, x):
        if self.has_bias:
            # expand bias to third dimension in order order to make it addible to result tensor
            expanded_bias = self.bias.unsqueeze(1).expand(-1, x.size(1), -1)
            return torch.add(torch.bmm(x, self.W), expanded_bias)

        return torch.bmm(x, self.W)




