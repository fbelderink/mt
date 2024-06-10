import torch
import numpy as np
import torch.nn as nn
from preprocessing.dataset import TranslationDataset
from preprocessing.dictionary import Dictionary
from preprocessing.fragment import create_source_window_matrix
from typing import List


def translate(model: nn.Module,
              source_data: List[List[str]],
              source_dict: Dictionary,
              target_dict: Dictionary,
              beam_size: int,
              window_size: int):
    model.eval()

    source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)

    get_target_idx = np.vectorize(target_dict.get_index_of_string)
    get_target_string = np.vectorize(target_dict.get_string_at_index)

    target_sentences = []

    for sentence in source_data:
        S = create_source_window_matrix(sentence, source_dict, window_size, len(sentence) * 1 + 1)

        S = torch.from_numpy(S)

        beam_targets = torch.from_numpy(get_target_idx([['<s>'] * window_size]))

        top_k_values = [0]
        top_k_indices = [[target_dict.get_index_of_string("<s>")] * window_size] * beam_size

        for s in S:
            # expand to make first dimension fit beam_size
            s = s.expand((beam_targets.shape[0], *s.shape))

            # predict on all previous top k results simultaneously
            pred = model(s, beam_targets)

            # add previous top k values along beam size dim
            pred += torch.tensor(top_k_values).unsqueeze(1)

            # flatten predictions to get top k along all previous top k predictions
            pred = pred.reshape((1, beam_targets.shape[0] * pred.shape[-1]))

            # get top k predictions
            top_k = pred.topk(beam_size, dim=-1)

            if beam_targets.shape[0] == 1:
                # only applies in first iteration, when beam targets have one dim only
                beam_targets = beam_targets.repeat(beam_size, 1)

            # init list to record new top k predictions
            new_top_k_indices = []
            for i in range(beam_size):

                # we flattened the prediction vector, so find out to which previous predictions new top k indices belong
                previous_indices = [i // len(target_dict) for i in top_k.indices.squeeze(0).tolist()]

                # get index in target vocab range (also has to happen because of flattening)
                current_indices = [i % len(target_dict) for i in top_k.indices.squeeze(0).tolist()]

                # select correct previous list and add new index
                new_top_k_indices.append(top_k_indices[previous_indices[i]] + [current_indices[i]])

                # replace indices in input for net
                beam_targets[i] = torch.tensor(new_top_k_indices[i][-window_size:])

            # record current best values
            top_k_values = top_k.values.squeeze(0).tolist()
            # replace top k indices whit newly found best indices
            top_k_indices = new_top_k_indices

        # get target translation (first window_size entries are sos)
        target_sentences.append([sentence[window_size:] for sentence in get_target_string(top_k_indices).tolist()])

    return target_sentences
