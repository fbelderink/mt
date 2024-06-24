from preprocessing.dictionary import Dictionary
import torch.nn as nn
from typing import List

from search.beam_search import translate as beam_translate


def translate(model: nn.Module,
              source_data: List[List[str]],
              source_dict: Dictionary,
              target_dict: Dictionary,
              window_size: int,
              alignment_factor=1):
    return beam_translate(model, source_data, source_dict, target_dict, 1, window_size, False, alignment_factor)
