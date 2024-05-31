import torch
from torch.utils.data import Dataset
from preprocessing.dictionary import Dictionary
from preprocessing.fragment import fragment_data_to_indices
from typing import List


class TranslationDataset(Dataset):
    def __init__(self, source_data: List[List[str]], target_data: List[List[str]],
                 source_dict: Dictionary, target_dict: Dictionary, window_size: int):
        filtered_source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)
        filtered_target_data = target_dict.apply_vocabulary_to_text(target_data, bpe_performed=False)

        S, T, L = fragment_data_to_indices(filtered_source_data, filtered_target_data, window_size, source_dict,
                                           target_dict)

        self._source_window_mat = torch.from_numpy(S)
        self._target_window_mat = torch.from_numpy(T)
        self._labels = torch.from_numpy(L)

        self._source_dict_size = len(source_dict)
        self._target_dict_size = len(target_dict)

        self._window_size = window_size

    def __len__(self):
        return self._source_window_mat.shape[0]

    def __getitem__(self, idx):
        return self._source_window_mat[idx, :], self._target_window_mat[idx, :], self._labels[idx, :]

    @staticmethod
    def load(path):
        return torch.load(path)

    def save(self, path):
        torch.save(self, path)

    def get_source_dict_size(self):
        return self._source_dict_size

    def get_target_dict_size(self):
        return self._target_dict_size

    def get_window_size(self):
        return self._window_size
