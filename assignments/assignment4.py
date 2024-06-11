import torch.nn as nn
from search.beam_search import translate
from search.greedy_search import translate as greedy_translate
from typing import List
from preprocessing.dictionary import Dictionary
from utils.file_manipulation import save_n_best_translations, save_data
from preprocessing.postprocessing import undo_prepocessing
from scoring.score import get_scores
import numpy as np


def test_beam_search(model: nn.Module, source_data: List[List[str]], source_dict: Dictionary,
                     target_dict: Dictionary, beam_size: int, window_size: int):
    target_sentences = translate(model, source_data, source_dict, target_dict, beam_size, window_size)

    post_processed_sentences = []
    for sentence in target_sentences:
        post_processed_sentences.append(undo_prepocessing(sentence))

    save_n_best_translations("eval/translations/beam_translations", post_processed_sentences)


def test_greedy_search(model: nn.Module, source_data: List[List[str]], source_dict: Dictionary, target_dict: Dictionary,
                       window_size: int):
    target_sentences = greedy_translate(model, source_data, source_dict, target_dict, window_size)

    post_processed_sentences = undo_prepocessing(target_sentences)

    save_data("eval/translations/greedy_translations", post_processed_sentences)


def test_get_scores(model: nn.Module, source_data: List[List[str]], target_data: List[List[str]],
                    source_dict: Dictionary, target_dict: Dictionary, window_size: int):
    scores = get_scores(model, source_data, target_data, source_dict, target_dict, window_size)

    print(np.max(scores))
    print(len(scores))
