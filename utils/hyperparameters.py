import torch


class Hyperparameters:
    def __init__(self, config):
        self.config = config
        self.dimensions = [x for x in list(config["dimensions"].values())]
        self.activation = getattr(torch.nn.functional, config["activation_function"])
        self.optimizer = getattr(torch.optim, config["optimizer"])
        self.learning_rate = config["learning_rate"]
        self.use_custom_ll = bool(config["use_custom_linear_layer"])
        self.batch_size = config["batch_size"]
        self.checkpoints = config["checkpoints_per_epoch"]
        self.saved_model = config["load_model_path"]
        self.half_lr = config["half_lr"]
        self.window_size = config["window_size"]
        self.dropout_rate = config["dropout_rate"]
        self.early_stopping = config["early_stopping"]
