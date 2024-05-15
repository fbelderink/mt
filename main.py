import argparse
from utils.file_manipulation import load_data
from assignments.assignment2 import task_batches, task_evaluate


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

    task_evaluate(multi30k_de, multi30k_en)
    task_batches(args.window_size, args.batch_size, multi30k_de, multi30k_en)
