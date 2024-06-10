import torch
import numpy as np
import torch.nn as nn
from preprocessing.dataset import TranslationDataset
from preprocessing.dictionary import Dictionary
from preprocessing.fragment import fragment_data_to_indices
from typing import List


def translate(model: nn.Module, source_data: List[List[str]], target_data: List[List[str]],
              source_dict: Dictionary, target_dict: Dictionary, beam_size: int):
    window_size = 2
    model.eval()

    source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)

    get_target_idx = np.vectorize(target_dict.get_index_of_string)

    for sentence in source_data:
        S, _, _ = fragment_data_to_indices([sentence], target_data, window_size, source_dict, target_dict)

        current_target = torch.from_numpy([get_target_idx(['<s>'] * window_size)] * beam_size)

        print(current_target.shape)

        """
        for s in S:
            #s = s.unsqueeze(0)
            #current_target = current_target.unsqueeze(0)

            pred = model(s, current_target)
            top_k_indices = pred.topk(beam_size)
        """







