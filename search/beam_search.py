import torch
import numpy as np
import torch.nn as nn
from preprocessing.dictionary import Dictionary
from preprocessing.batching.fragment import create_source_window_matrix
from typing import List


def translate(model: nn.Module,
              source_data: List[List[str]],
              source_dict: Dictionary,
              target_dict: Dictionary,
              beam_size: int,
              window_size: int,
              get_n_best=False,
              alignment_factor=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)

    get_target_idx = np.vectorize(target_dict.get_index_of_string)
    get_target_string = np.vectorize(target_dict.get_string_at_index)

    target_sentences = []

    for sentence in source_data:
        S = create_source_window_matrix(sentence, source_dict, window_size, len(sentence) * alignment_factor + 1)

        S = torch.from_numpy(S).to(device)

        beam_targets = torch.from_numpy(get_target_idx([['<s>'] * window_size])).to(device)

        top_k_values = [0]
        top_k_indices = [[target_dict.get_index_of_string("<s>")] * window_size] * beam_size

        for s in S:
            # expand to make first dimension fit beam_size
            s = s.expand((beam_targets.shape[0], *s.shape))

            # predict on all previous top k results simultaneously
            pred = model(s, beam_targets)

            # add previous top k values along beam size dim
            pred += torch.tensor(top_k_values).unsqueeze(1).to(device)

            # flatten predictions to get top k along all previous top k predictions
            # (1, beam_size * vocab_size)
            pred = pred.reshape((1, beam_targets.shape[0] * pred.shape[-1]))

            # get top k predictions
            top_k = pred.topk(beam_size, dim=-1)

            if beam_targets.shape[0] == 1:
                # only applies in first iteration, when beam targets have one dim only
                beam_targets = beam_targets.repeat(beam_size, 1)

            # we flattened the prediction vector, so find out to which previous predictions new top k indices belong
            previous_indices = [i // len(target_dict) for i in top_k.indices.squeeze(0).tolist()]

            # get indices in target vocab range (also has to happen because of flattening)
            current_indices = [i % len(target_dict) for i in top_k.indices.squeeze(0).tolist()]

            # init list to record new top k predictions
            new_top_k_indices = []

            for i in range(beam_size):
                # select correct previous list and add new index
                new_top_k_indices.append(top_k_indices[previous_indices[i]] + [current_indices[i]])

                # replace indices in input for net
                beam_targets[i] = torch.tensor(new_top_k_indices[i][-window_size:]).to(device)

            # record current best values
            top_k_values = top_k.values.squeeze(0).tolist()
            # replace top k indices with newly found best indices
            top_k_indices = new_top_k_indices

        # get target translation (first window_size entries are sos)
        if not get_n_best:
            target_sentences.append(get_target_string(top_k_indices[np.argmax(top_k_values)][window_size:]).tolist())
        else:
            target_sentences.append(get_target_string([indices[window_size:] for indices in top_k_indices]).tolist())

    return target_sentences
