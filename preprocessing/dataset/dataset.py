from __future__ import annotations
import torch
from torch.utils.data import Dataset
from preprocessing.dictionary import Dictionary
from preprocessing.batching.fragment import fragment_data_to_indices
from typing import List
import numpy as np
import sys
import os


class TranslationDataset(Dataset):
    def __init__(self, source_dict: Dictionary, target_dict: Dictionary):
        super(TranslationDataset, self).__init__()
        self._source_dict_size = len(source_dict)
        self._target_dict_size = len(target_dict)
        self._eos_idx = target_dict.get_index_of_string('</s>')

    @staticmethod
    def load(path):
        sys.path.append(os.path.join(os.path.abspath(os.path.curdir), 'preprocessing'))
        return torch.load(path)

    def save(self, path):
        torch.save(self, path)

    def get_source_dict_size(self):
        return self._source_dict_size

    def get_target_dict_size(self):
        return self._target_dict_size

    def get_eos_idx(self):
        return self._eos_idx


class FFTranslationDataset(TranslationDataset):
    def __init__(self, source_data: List[List[str]], target_data: List[List[str]],
                 source_dict: Dictionary, target_dict: Dictionary, window_size: int):
        super(FFTranslationDataset, self).__init__(source_dict, target_dict)

        filtered_source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)
        filtered_target_data = target_dict.apply_vocabulary_to_text(target_data, bpe_performed=False)

        S, T, L = fragment_data_to_indices(filtered_source_data, filtered_target_data, window_size, source_dict,
                                           target_dict)

        self._source_window_mat = torch.from_numpy(S)
        self._target_window_mat = torch.from_numpy(T)
        self._labels = torch.from_numpy(L)

        self._window_size = window_size

    def __len__(self):
        return self._source_window_mat.shape[0]

    def __getitem__(self, idx):
        return self._source_window_mat[idx, :], self._target_window_mat[idx, :], self._labels[idx, :]

    def get_window_size(self):
        return self._window_size


class RNNTranslationDataset(TranslationDataset):
    def __init__(self, source_data: List[List[str]], target_data: List[List[str]],
                 source_dict: Dictionary, target_dict: Dictionary):
        super(RNNTranslationDataset, self).__init__(source_dict, target_dict)

        filtered_source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)
        filtered_target_data = target_dict.apply_vocabulary_to_text(target_data, bpe_performed=False)

        self._source_dict_size = len(source_dict)
        self._target_dict_size = len(target_dict)

        T_max = len(max(filtered_source_data + filtered_target_data, key=lambda sentence: len(sentence)))

        filtered_source_data = [sentence + ['</s>'] * (T_max - len(sentence)) for sentence in filtered_source_data]
        filtered_target_data = [['<s>'] + sentence + ['</s>'] * (T_max - len(sentence)) for sentence in
                                filtered_target_data]

        get_source_index = np.vectorize(source_dict.get_index_of_string)
        get_target_index = np.vectorize(target_dict.get_index_of_string)

        self._source_data = torch.from_numpy(get_source_index(np.array(filtered_source_data)))
        self._target_data = torch.from_numpy(get_target_index(np.array(filtered_target_data)))
        self._labels = torch.from_numpy(get_target_index(np.array([sentence[1:] for sentence in filtered_target_data])))

    def __len__(self):
        return len(self._source_data)

    def __getitem__(self, idx):
        return self._source_data[idx], self._target_data[idx], self._labels[idx]

