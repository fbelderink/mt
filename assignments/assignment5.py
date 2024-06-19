from typing import List
from preprocessing.dataset import RNNTranslationDataset
from preprocessing.BPE import generate_bpe
from preprocessing.dictionary import Dictionary


def generate_dataset(source_data: List[List[str]],
                     target_data: List[List[str]],
                     num_operations,
                     dict_de=None,
                     dict_en=None,
                     save_path=None) -> RNNTranslationDataset:
    if not dict_de:
        german_ops = generate_bpe(source_data, num_operations)
        dict_de = Dictionary(target_data, german_ops)
        dict_de.save('data/dicts/train_dict_de.pkl')

    if not dict_en:
        english_ops = generate_bpe(source_data, num_operations)
        dict_en = Dictionary(target_data, english_ops)
        dict_en.save('data/dicts/train_dict_en.pkl')

    dataset = RNNTranslationDataset(source_data, target_data, dict_de, dict_en)

    if save_path is not None:
        dataset.save(save_path)

    return dataset
