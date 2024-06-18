from preprocessing.dictionary import Dictionary
import torch
import torch.nn as nn
import numpy as np
from typing import List
from preprocessing.fragment import create_source_window_matrix

from search.beam_search import translate as beam_translate


def translate(model: nn.Module,
              source_data: List[List[str]],
              source_dict: Dictionary,
              target_dict: Dictionary,
              window_size: int,
              alignment_factor=1):
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()

    # preprocessed source sentences   
    source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)

    target_sentences = []

    for sentence in source_data:
        # gives the model more flexibility on when to end the sentence
        number_eos = 5
        for i in range(number_eos):
            sentence.append('</s>')

        # contains all entries needed to translate current sentence
        S = create_source_window_matrix(sentence, source_dict, window_size, len(sentence) * alignment_factor + 1)
        S = torch.from_numpy(S)

        # contains the entry needed to translate the next sentence 
        t_list = ['<s>'] * window_size

        get_target_idx = np.vectorize(target_dict.get_index_of_string)
        target = torch.from_numpy(get_target_idx(t_list))  # replace tokens with corresponding indices

        translated_sentence = []

        for s in S:
            if len(translated_sentence) > 0 and translated_sentence[-1] == '</s>':
                break

            s = s.unsqueeze(0)
            target = target.unsqueeze(0)

            s = s.to(device)
            target = target.to(device)

            pred = model(s, target)
            res = target_dict.get_string_at_index(int(torch.argmax(pred)))
            translated_sentence.append(res)

            # update t
            t_list.append(res)
            t_list.pop(0)

            target = torch.from_numpy(get_target_idx(t_list))

        target_sentences.append(translated_sentence)

    return target_sentences
    """
    return beam_translate(model, source_data, source_dict, target_dict, 1, window_size, False, alignment_factor)
