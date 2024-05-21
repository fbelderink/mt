import argparse
from utils.file_manipulation import load_data
from assignments.assignment3 import *


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

    # generate dataset
    #generate_dataset(multi30k_de, multi30k_en, args.window_size, 7000, save_path='data/train7k.pt')
    test_dataset_load('data/train7k.pt')
