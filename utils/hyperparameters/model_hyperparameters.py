import torch


class ModelHyperparameters:
    def __init__(self, config):
        self.config = config


class FFModelHyperparameters(ModelHyperparameters):
    def __init__(self, config):
        super(FFModelHyperparameters, self).__init__(config)

        # model config
        self.dimensions = [x for x in list(config["dimensions"].values())]
        self.activation = getattr(torch.nn.functional, config["activation_function"])
        self.use_custom_ll = bool(config["use_custom_linear_layer"])

        # train config
        self.dropout_rate = config["dropout_rate"]


class RNNModelHyperparameters(ModelHyperparameters):
    def __init__(self, config):
        super(RNNModelHyperparameters, self).__init__(config)

        self.encoder_parameters = [x for x in list(config["encoder_parameters"].values())]
        self.decoder_parameters = [x for x in list(config["decoder_parameters"].values())]

        self.lstm_layers = config["lstm_layers"]
        self.lstm_hidden_dim = config["lstm_hidden_dim"]
        self.lstm_bidirectional = config["lstm_bidirectional"]
