import argparse
from utils.file_manipulation import load_data
from assignments.assignment1 import *
from assignments.assignment3 import *
from assignments.assignment2 import *
from assignments.assignment4 import *
import training.train as train
from utils.ConfigLoader import ConfigLoader
from utils.hyperparameters import Hyperparameters
import pickle
from metrics.metrics import BLEU


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
    parser.add_argument('-bs', '--do_bleu_search', type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    """
    hyps = load_data(args.hyps)
    refs = load_data(args.refs)

    first_assignment(args, hyps, refs)
    """

    data_de = load_data(args.data_de_path)
    data_en = load_data(args.data_en_path)

    data_de_dev = load_data(args.data_de_dev_path)
    data_en_dev = load_data(args.data_en_dev_path)

    dict_de = Dictionary.load("data/dicts/train_dict_de.pkl")
    dict_en = Dictionary.load("data/dicts/train_dict_en.pkl")

    #generate dataset
    #generate_dataset(data_de, data_en, args.window_size, 7000,
    #                 dict_de=dict_de, dict_en=dict_en, save_path=f'data/train7k_w{args.window_size}.pt')
    #test_dataset_load(f'data/train7k_w{args.window_size}.pt')

    #generate_dataset(data_de_dev, data_en_dev, args.window_size, 7000,
    #                 dict_de=dict_de, dict_en=dict_en, save_path=f'data/val7k_w{args.window_size}.pt')
    #test_dataset_load(f'data/val7k_w{args.window_size}.pt')

    #train.train("data/train7k.pt", "data/val7k.pt",
    #            Hyperparameters(ConfigLoader("configs/config.yaml").get_config()), val_rate=1)

    #model = torch.load("eval/checkpoints/12-06-2024/21_19_51.pth", map_location=torch.device("cpu"))

    #translations = load_data("eval/translations/beam_translations")

    #translations = test_beam_search(model, de_data, dict_de, dict_en, 3, args.window_size)
    #translations = test_greedy_search(model, data_de, dict_de, dict_en, args.window_size)
    #test_get_scores(model, source_data, target_data, dict_de, dict_en, args.window_size)
    #test_model_bleu(model, data_de, data_en, dict_de, dict_en,
    #                3, args.window_size, args.do_bleu_search, None)

    bleus = determine_models_bleu('eval/checkpoints/12-06-2024', data_de_dev, data_en_dev, dict_de, dict_en,
                                  3, 2, True)

    print(f"Best model: {max(bleus)}")
