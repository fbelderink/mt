import argparse
from utils.file_manipulation import load_data, save_checkpoint
from typing import List
from preprocessing.BPE import generate_bpe
from preprocessing.dictionary import Dictionary
from dataset import RNNTranslationDataset, FFTranslationDataset


def _create_dict(data: List[List[str]],
                 num_operations,
                 save_path) -> Dictionary:
    print("started creating dictionary")
    ops = generate_bpe(data, num_operations)
    dictionary = Dictionary(data, ops)
    print("finished creating dictionary")
    if save_path:
        dictionary.save(save_path)
        print(f"saved dictionary at {save_path}")
    return dictionary


def _generate_dataset_rnn(source_data: List[List[str]],
                          target_data: List[List[str]],
                          num_operations: int,
                          dict_de=None,
                          dict_de_save_path=None,
                          dict_en=None,
                          dict_en_save_path=None,
                          save_path=None) -> (RNNTranslationDataset, Dictionary, Dictionary):
    if not dict_de:
        dict_de = _create_dict(source_data, num_operations, dict_de_save_path)

    if not dict_en:
        dict_en = _create_dict(target_data, num_operations, dict_en_save_path)

    dataset = RNNTranslationDataset(source_data, target_data, dict_de, dict_en)

    if save_path is not None:
        dataset.save(save_path)

    return dataset, dict_de, dict_en


def _generate_dataset_ff(source_data: List[List[str]],
                         target_data: List[List[str]],
                         num_operations: int,
                         window_size: int,
                         dict_de=None,
                         dict_de_save_path=None,
                         dict_en=None,
                         dict_en_save_path=None,
                         save_path=None) -> (FFTranslationDataset, Dictionary, Dictionary):
    if not dict_de:
        dict_de = _create_dict(source_data, num_operations, None, dict_de_save_path)

    if not dict_en:
        dict_en = _create_dict(target_data, num_operations, None, dict_en_save_path)

    dataset = FFTranslationDataset(source_data, target_data, dict_de, dict_en, window_size)

    if save_path is not None:
        dataset.save(save_path)

    return dataset, dict_de, dict_en


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-de', '--data_de_path', type=str)
    parser.add_argument('-en', '--data_en_path', type=str)
    parser.add_argument('-de_dev', '--data_de_dev_path', type=str)
    parser.add_argument('-en_dev', '--data_en_dev_path', type=str)
    parser.add_argument('-edp', '--dict_en_path', type=str)
    parser.add_argument('-ddp', '--dict_de_path', type=str)
    parser.add_argument('-tdsp_rnn', '--train_dataset_save_path_rnn', type=str)
    parser.add_argument('-vdsp_rnn', '--val_dataset_save_path_rnn', type=str)
    parser.add_argument('-tdsp_ff', '--train_dataset_save_path_ff', type=str)
    parser.add_argument('-vdsp_ff', '--val_dataset_save_path_ff', type=str)
    parser.add_argument('-bpe_ops', '--bpe_operations', type=int)
    parser.add_argument('-w', '--window_size', type=int)
    parser.add_argument('-rnn', '--gen_rnn', type=bool)
    parser.add_argument('-ff', '--gen_ff', type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    data_de = load_data(args.data_de_path)
    data_en = load_data(args.data_en_path)

    data_de_dev = load_data(args.data_de_dev_path)
    data_en_dev = load_data(args.data_en_dev_path)

    dict_de, dict_en = None, None

    if args.dict_de_path:
        try:
            dict_de = Dictionary.load(args.dict_de_path)
        except FileNotFoundError:
            pass
    if args.dict_en_path:
        try:
            dict_en = Dictionary.load(args.dict_en_path)
        except FileNotFoundError:
            pass

    if args.gen_rnn:
        print("start generating rnn training dataset")
        _, dict_de, dict_en = _generate_dataset_rnn(data_de, data_en, args.bpe_operations,
                              dict_de, args.dict_de_path,
                              dict_en, args.dict_en_path,
                              args.train_dataset_save_path_rnn)
        print("finished generating rnn training dataset")

        print("start generating rnn validation dataset")
        _generate_dataset_rnn(data_de_dev, data_en_dev, args.bpe_operations,
                              dict_de, args.dict_de_path,
                              dict_en, args.dict_en_path,
                              args.val_dataset_save_path_rnn)
        print("finished generating rnn validation dataset")

    if args.gen_ff:
        print("start generating ff training dataset")
        _, dict_de, dict_en = _generate_dataset_ff(data_de, data_en, args.bpe_operations, args.window_size,
                             dict_de, args.dict_de_path,
                             dict_en, args.dict_en_path,
                             args.train_dataset_save_path_ff)
        print("finished generating ff training dataset")

        print("start generating ff validation dataset")
        _generate_dataset_ff(data_de_dev, data_en_dev, args.bpe_operations, args.window_size,
                             dict_de, args.dict_de_path,
                             dict_en, args.dict_en_path,
                             args.val_dataset_save_path_ff)
        print("finished generating ff validation dataset")
