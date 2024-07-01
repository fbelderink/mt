from search import beam_search
from search import greedy_search
import torch.nn as nn
from preprocessing.dictionary import Dictionary
from typing import List
from postprocessing.postprocessing import undo_prepocessing
from metrics.metrics import BLEU
from torchtext.data.metrics import bleu_score as torch_bleu
from model.ff.feedforward_net import FeedforwardNet
from model.seq2seq.recurrent_net import RecurrentNet


def get_bleu_of_model(model: nn.Module,
                      source_data: List[List[str]],
                      reference_data: List[List[str]],
                      source_dict: Dictionary,
                      target_dict: Dictionary,
                      beam_size: int,
                      window_size: int,
                      do_beam_search: bool,
                      translations: List[List[str]] = None,
                      use_torch_bleu=False):
    bleu = BLEU()

    if translations is not None:
        if use_torch_bleu:
            return torch_bleu(translations, [[ref] for ref in reference_data])
        return bleu(translations, reference_data)

    if isinstance(model, FeedforwardNet):
        if do_beam_search:
            translations = beam_search.translate_ff(model,
                                                    source_data,
                                                    source_dict,
                                                    target_dict,
                                                    beam_size,
                                                    window_size)
        else:
            translations = greedy_search.translate_ff(model,
                                                      source_data,
                                                      source_dict,
                                                      target_dict,
                                                      window_size)
    elif isinstance(model, RecurrentNet):
        if do_beam_search:
            translations = beam_search.translate_rnn(model, source_data, source_dict, target_dict, beam_size)
        else:
            translations = greedy_search.translate_rnn(model,
                                                       source_data,
                                                       source_dict,
                                                       target_dict)
    else:
        raise ValueError('Unsupported model')

    translation = undo_prepocessing(translations)

    if use_torch_bleu:
        bleu_score = torch_bleu(translation, reference_data)
    else:
        bleu_score = bleu(translation, reference_data)

    return bleu_score
