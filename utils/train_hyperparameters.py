import torch


class TrainHyperparameters:
    def __init__(self, config):
        self.config = config

        # train config
        self.optimizer = getattr(torch.optim, config["optimizer"])
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.checkpoints = config["checkpoints_per_epoch"]
        self.saved_model = config["load_model_path"]
        self.half_lr = config["half_lr"]
        self.early_stopping = config["early_stopping"]
        self.shuffle = config["shuffle"]
        self.num_workers = config["num_workers"]
        self.max_epochs = config["max_epochs"]
        self.print_eval_every = config["print_eval_every"]
        self.test_model_every = config["test_model_every"]


class RNNTrainHyperparameters(TrainHyperparameters):
    def __init__(self, config):
        super(RNNTrainHyperparameters, self).__init__(config)

        self.two_optimizers = config["two_optimizers"]
        self.teacher_forcing = config["teacher_forcing"]
        self.ignore_eos_for_acc = config["ignore_eos_for_acc"]


class FFTrainHyperparameters(TrainHyperparameters):
    def __init__(self, config):
        super(FFTrainHyperparameters, self).__init__(config)
