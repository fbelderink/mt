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

    dict_de = Dictionary.load("data/dicts/train_dict_de.pkl")
    dict_en = Dictionary.load("data/dicts/train_dict_en.pkl")

    #generate dataset
    #generate_dataset(multi30k_de, multi30k_en, args.window_size, 7000,
    #                 dict_de=dict_de, dict_en=dict_en, save_path='data/val7k.pt')
    #test_dataset_load('data/train7k.pt')

    #train.train("data/train7k.pt", "data/val7k.pt",
    #            Hyperparameters(ConfigLoader("configs/config.yaml").get_config()))

    model = torch.load("eval/checkpoints/17_16_39.pth")

    translations = load_data("eval/translations/beam_translations")

    #translations = test_beam_search(model, de_data, dict_de, dict_en, 3, args.window_size)
    #translations = test_greedy_search(model, data_de, dict_de, dict_en, args.window_size)
    #test_get_scores(model, source_data, target_data, dict_de, dict_en, args.window_size)
    test_model_bleu(model, data_de, data_en, dict_de, dict_en,
                    3, args.window_size, args.do_bleu_search, translations)
