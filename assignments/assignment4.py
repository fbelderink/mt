import torch.nn as nn
from search.beam_search import translate
from typing import List
from preprocessing.dictionary import Dictionary

def test_beam_search(model: nn.Module, source_data: List[List[str]], target_data: List[List[str]],
              source_dict: Dictionary, target_dict: Dictionary, beam_size: int):
    translate(model, source_data, target_data, source_dict, target_dict, beam_size)
