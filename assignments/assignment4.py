import torch.nn as nn
from search.beam_search import translate
from typing import List
from preprocessing.dictionary import Dictionary
from utils.file_manipulation import save_n_best_translations
from preprocessing.postprocessing import undo_prepocessing


def test_beam_search(model: nn.Module, source_data: List[List[str]], source_dict: Dictionary,
                     target_dict: Dictionary, beam_size: int, window_size: int):
    target_sentences = translate(model, source_data, source_dict, target_dict, beam_size, window_size)

    post_processed_sentences = []
    for sentence in target_sentences:
        post_processed_sentences.append(undo_prepocessing(sentence))

    save_n_best_translations("eval/translations/n_best_translations", post_processed_sentences)


def test_greedy_search():
    pass
