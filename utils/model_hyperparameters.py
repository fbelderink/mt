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

        self.rnn_type = config["rnn_type"]
        self.rnn_layers = config["rnn_layers"]
        self.rnn_hidden_dim = config["rnn_hidden_dim"]
        self.use_attention = config["use_attention"]
        self.use_attention_dp = config["use_attention_dp"]
        self.use_attention_mask = config["use_attention_mask"]
        self.attention_dropout = config["attention_dropout"]
        self.hidden_ll = config["ll_hidden_dim"]
        self.num_ll = config["num_ll"]
        self.dropout_ll = config["dropout_ll"]
        self.batch_norm_ll = config["batch_norm_ll"]
        self.activation_function_ll = config["activation_function_ll"]
