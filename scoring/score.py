from assignments.assignment3 import *
import torch
import numpy as np

from typing import List
import torch.nn as nn
from preprocessing.fragment import fragment_data_to_indices


def get_scores(model: nn.Module, source_data: List[List[str]], target_data: List[List[str]],
               source_dict: Dictionary, target_dict: Dictionary, window_size: int):
    source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)
    target_data = target_dict.apply_vocabulary_to_text(target_data, bpe_performed=False)

    model.eval()

    scores = []

    for source_sentence, target_sentence in zip(source_data, target_data):
        log_sum = 0
        S, T, L = fragment_data_to_indices([source_sentence], [target_sentence],
                                           window_size, source_dict, target_dict)

        S = torch.from_numpy(S)
        T = torch.from_numpy(T)
        L = torch.from_numpy(L)

        for s, t, l in zip(S, T, L):
            s = s.unsqueeze(0)
            t = t.unsqueeze(0)

            pred = model(s, t)

            log_sum += pred.squeeze(0)[l.item()].item() / len(target_sentence)

        scores.append(np.exp(log_sum))

    return scores

