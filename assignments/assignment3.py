import torch

from model.layers.linear import LinearLayer


def test_linear_layer():
    layer = LinearLayer(3, 3, 5, bias=False)

    torch.manual_seed(0)
    x = torch.randn(3, 3, 1)

    print(layer(x))
