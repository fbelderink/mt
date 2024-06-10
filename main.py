import argparse
from utils.file_manipulation import load_data
from assignments.assignment3 import *
from assignments.assignment2 import *
from assignments.assignment4 import *
import training.train as train
from utils.ConfigLoader import ConfigLoader
from utils.hyperparameters import Hyperparameters
import pickle


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-hy', '--hyps', type=str)
    parser.add_argument('-r', '--refs', type=str)
    parser.add_argument('-B', '--batch_size', type=int)
    parser.add_argument('-w', '--window_size', type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    multi30k_de = load_data(args.hyps)
    multi30k_en = load_data(args.refs)

    #task_batches(args.window_size, args.batch_size, multi30k_de, multi30k_en)

    dict_de = Dictionary.load("data/dicts/train_dict_de.pkl")
    dict_en = Dictionary.load("data/dicts/train_dict_en.pkl")

    #generate dataset
    #generate_dataset(multi30k_de, multi30k_en, args.window_size, 7000,
    #                 dict_de=dict_de, dict_en=dict_en, save_path='data/val7k.pt')
    #test_dataset_load('data/train7k.pt')

    #train.train("data/train7k.pt", "data/val7k.pt",
    #            Hyperparameters(ConfigLoader("configs/config.yaml").get_config()))

    model = torch.load("eval/checkpoints/10-06-2024/16_00_38.pth")

    test_beam_search(model, multi30k_de[0:100], dict_de, dict_en, 3, args.window_size)
