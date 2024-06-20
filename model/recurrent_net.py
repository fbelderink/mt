import torch.nn as nn
from model.basic_net import BasicNet
from utils.hyperparameters import RNNHyperparameters


class RecurrentNet(BasicNet):
    def __init__(self, source_dict_size, target_dict_size, config: RNNHyperparameters, model_name="rnn"):
        super(RecurrentNet, self).__init__(source_dict_size, target_dict_size, config, model_name)

        #hyperparameters
        self.embed_dim = config.encoder_parameters[0]
        self.encoder_hidden_dim = config.encoder_parameters[1]
        self.encoder_layers = config.encoder_parameters[2]
        self.encoder_dropout = config.encoder_parameters[3]

        self.decoder_hidden_dim = config.decoder_parameters[0]
        self.decoder_layers = config.decoder_parameters[1]
        self.decoder_dropout = config.decoder_parameters[2]

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
