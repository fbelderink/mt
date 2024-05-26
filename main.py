import argparse
from utils.file_manipulation import load_data
from assignments.assignment3 import *
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
  #  with open("eval/dict_de.pkl", 'rb') as f:
   #     dict_de = pickle.load(f)
    #with open("eval/dict_en.pkl", 'rb') as f:
     #   dict_en = pickle.load(f)
    # generate dataset
    generate_dataset(multi30k_de, multi30k_en, args.window_size, 7000, save_path='data/train100s', dict_de=None, dict_en=None)
    test_dataset_load('data/train100s')
    train.train("data/train100s", None, Hyperparameters(ConfigLoader("utils/config.yaml").get_config()))