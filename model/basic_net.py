import torch.nn as nn
import abc
from utils.hyperparameters import Hyperparameters


class BasicNet(nn.Module):

    def __init__(self, source_dict_size, target_dict_size,
                 config: Hyperparameters, model_name=""):
        super(BasicNet, self).__init__()

        self.source_dict_size = source_dict_size
        self.target_dict_size = target_dict_size

        self.model_name = model_name

    @abc.abstractmethod
    def forward(self, S, T):
        pass

    @abc.abstractmethod
    def compute_loss(self, pred, label):
        pass

    def print_structure(self):
        print(f"model name: {self.model_name}")
        print(f"total number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        print("Layers:")
        for name, module in self.named_children():
            if not isinstance(module, nn.CrossEntropyLoss):
                print(f"{name}: {module}")
