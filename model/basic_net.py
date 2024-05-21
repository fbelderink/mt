import torch.nn as nn


class BasicNet(nn.Module):
    def __init__(self, replace_linear_layer=False):
        super(BasicNet, self).__init__()
        """
        TODO: 
        - Modell anlegen
        - replace linear layers optionally
        """
        pass


    def forward(self, x):
        pass

    def compute_loss(self, pred, label):
        pass

    def print_structure(self):
        """
        TODO:
        vor Beginn des Trainings die Gesamtzahl der trainierbaren Parameter sowie die ver-
        wendeten shapes von allen verwendeten Tensoren (also die Dimensionalität deren Aus-
        gabe) zusammen mit dem entsprechenden Namen der Layer auszugeben und damit
        die Netzwerkstruktur deutlich machen.

        """
        pass
