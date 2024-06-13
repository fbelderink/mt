from search import beam_search
from search import greedy_search
import torch.nn as nn
from preprocessing.dictionary import Dictionary
from typing import List
from postprocessing.postprocessing import undo_prepocessing
from metrics.metrics import BLEU


def get_bleu_of_model(model: nn.Module,
                      source_data: List[List[str]],
                      reference_data: List[List[str]],
                      source_dict: Dictionary,
                      target_dict: Dictionary,
                      beam_size: int,
                      window_size: int,
                      do_beam_search: bool,
                      translations: List[List[str]] = None):
    bleu = BLEU()

    if translations is not None:
        return bleu(reference_data, translations)

    if do_beam_search:
        translations = beam_search.translate(model, source_data, source_dict, target_dict, beam_size, window_size)
    else:
        translations = greedy_search.translate(model, source_data, source_dict, target_dict, window_size)

    translation = undo_prepocessing(translations)

    bleu_score = bleu(translation, reference_data)

    return bleu_score
