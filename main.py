import argparse
from assignments.assignment4 import *
from utils.ConfigLoader import ConfigLoader
from utils.model_hyperparameters import RNNModelHyperparameters
from utils.train_hyperparameters import RNNTrainHyperparameters
from training.train import train


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-de', '--data_de_path', type=str)
    parser.add_argument('-en', '--data_en_path', type=str)
    parser.add_argument('-de_dev', '--data_de_dev_path', type=str)
    parser.add_argument('-en_dev', '--data_en_dev_path', type=str)
    parser.add_argument('-hy', '--hyps', type=str)
    parser.add_argument('-r', '--refs', type=str)
    parser.add_argument('-B', '--batch_size', type=int)
    parser.add_argument('-w', '--window_size', type=int)
    parser.add_argument('-bs', '--do_beam_search', type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    """
    hyps = load_data(args.hyps)
    refs = load_data(args.refs)

    first_assignment(args, hyps, refs)
    """

    # data_de = load_data(args.data_de_path)
    # data_en = load_data(args.data_en_path)

    # data_de_dev = load_data(args.data_de_dev_path)
    # data_en_dev = load_data(args.data_en_dev_path)

    dict_de = Dictionary.load("data/dicts/train_dict_de.pkl")
    dict_en = Dictionary.load("data/dicts/train_dict_en.pkl")

    model_config = ConfigLoader("./configs/rnn/config.yaml").get_config()

    train_config = ConfigLoader("./configs/training/rnn_train_config.yaml").get_config()

    model_params = RNNModelHyperparameters(model_config)

    train_params = RNNTrainHyperparameters(train_config)

    train(
        './data/train7k_rnn.pt',
        './data/val7k_rnn.pt',
        model_params,
        train_params
    )

