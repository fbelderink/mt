from preprocessing.dictionary import Dictionary
import torch.nn as nn
from typing import List
from model.basic_net import BasicNet

from search.beam_search import translate_ff as beam_translate_ff
from search.beam_search import translate_rnn as beam_translate_rnn


def translate_rnn(model: BasicNet,
                  source_data: List[List[str]],
                  source_dict: Dictionary,
                  target_dict: Dictionary):
    return beam_translate_rnn(model, source_data, source_dict, target_dict, 1, False)


def translate_ff(model: BasicNet,
                 source_data: List[List[str]],
                 source_dict: Dictionary,
                 target_dict: Dictionary,
                 window_size: int,
                 alignment_factor=1):
    return beam_translate_ff(model, source_data, source_dict, target_dict, 1, window_size, False, alignment_factor)
