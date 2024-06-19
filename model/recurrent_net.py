import torch
import torch.nn as nn
from utils.hyperparameters import RNNHyperparameters


class RecurrentNet(nn.Module):
    def __init__(self, source_dict_size, target_dict_size, config: RNNHyperparameters, model_name="rnn"):
        super(RecurrentNet, self).__init__()

        #hyperparameters
        self.embed_dim = config.encoder_parameters[0]
        self.encoder_hidden_dim = config.encoder_parameters[1]
        self.encoder_layers = config.encoder_parameters[2]
        self.encoder_dropout = config.encoder_parameters[3]

        self.model_name = model_name
        self.decoder_hidden_dim = config.decoder_parameters[0]
        self.decoder_layers = config.decoder_parameters[1]
        self.decoder_dropout = config.decoder_parameters[2]

        self.source_dict_size = source_dict_size
        self.target_dict_size = target_dict_size

        self.source_embedding = nn.Embedding(self.source_dict_size, self.embed_dim)

        self.encoder = nn.LSTM(self.embed_dim,
                               self.encoder_hidden_dim,
                               self.encoder_layers,
                               dropout=self.encoder_dropout,
                               batch_first=True)

        self.target_embedding = nn.Embedding(self.target_dict_size, self.embed_dim)
        self.decoder = nn.LSTM(self.embed_dim,
                               self.decoder_hidden_dim,
                               self.decoder_layers,
                               dropout=self.decoder_dropout,
                               batch_first=True)

    def forward(self, S, T):
        print(S.shape)
        src_embedded = self.source_embedding(S)
        print("src_embed", src_embedded.shape)
        encoder_output, _ = self.encoder(src_embedded)
        print("encoder", encoder_output.shape)

        target_embedded = self.target_embedding(T)
        decoder_output, _ = self.decoder(target_embedded)
        print("decoder", decoder_output.shape)

        return 0


    def compute_loss(self, pred, label):
        pass

    def print_structure(self):
        print("total number of trainable parameters: " + str(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print("Layers:")
        for name, module in self.named_children():
            if not isinstance(module, nn.CrossEntropyLoss):
                print(f"{name}: {module}")