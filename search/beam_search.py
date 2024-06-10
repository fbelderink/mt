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

    target_sentences = []

    for sentence in source_data:
        S, _, _ = fragment_data_to_indices([sentence], target_data, window_size, source_dict, target_dict)

        S = torch.from_numpy(S)

        beam_targets = torch.from_numpy(get_target_idx([['<s>'] * window_size]))

        top_k_values = [0]
        top_k_indices = [[target_dict.get_index_of_string("<s>")] * window_size] * beam_size

        get_target_string = np.vectorize(target_dict.get_string_at_index)

        for s in S:
            #expand to make first dimension fit beam_size
            s = s.expand((beam_targets.shape[0], *s.shape))

            pred = model(s, beam_targets)

            pred += torch.tensor(top_k_values).unsqueeze(1)

            pred = pred.reshape((1, beam_targets.shape[0] * pred.shape[-1]))

            top_k = pred.topk(beam_size, dim=-1)

            if beam_targets.shape[0] == 1:
                beam_targets = beam_targets.repeat(beam_size, 1)

            new_top_k_indices = []
            for i in range(beam_size):

                previous_indices = [i // len(target_dict) for i in top_k.indices.squeeze(0).tolist()]
                current_indices = [i % len(target_dict) for i in top_k.indices.squeeze(0).tolist()]

                new_top_k_indices.append(top_k_indices[previous_indices[i]] + [current_indices[i]])

                beam_targets[i] = torch.tensor(new_top_k_indices[i][-window_size:])

            top_k_values = top_k.values.squeeze(0).tolist()
            top_k_indices = new_top_k_indices

        target_sentences.append(get_target_string(top_k_indices))

        return target_sentences
