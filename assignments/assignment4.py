import torch
import torch.nn as nn
from search.beam_search import translate
from search.greedy_search import translate as greedy_translate
from typing import List
from preprocessing.dictionary import Dictionary
from utils.file_manipulation import save_data, save_n_best_translations
from postprocessing.postprocessing import undo_prepocessing
from metrics.calculate_bleu_of_model import get_bleu_of_model
from scoring.score import get_scores
import numpy as np
import os


def test_beam_search(model: nn.Module, source_data: List[List[str]], source_dict: Dictionary,
                     target_dict: Dictionary, beam_size: int, window_size: int, get_n_best=True):
    target_sentences = translate(model, source_data, source_dict, target_dict, beam_size, window_size, get_n_best)

    if get_n_best:
        post_processed_sentences = []
        for sentences in target_sentences:
            post_processed_sentences.append(undo_prepocessing(sentences))
    else:
        post_processed_sentences = undo_prepocessing(target_sentences)

    if not get_n_best:
        save_data("eval/translations/beam_translations", post_processed_sentences)
    else:
        save_n_best_translations("eval/translations/beam_translations", post_processed_sentences)

    return post_processed_sentences


def test_greedy_search(model: nn.Module, source_data: List[List[str]], source_dict: Dictionary, target_dict: Dictionary,
                       window_size: int):
    target_sentences = greedy_translate(model, source_data, source_dict, target_dict, window_size)

    post_processed_sentences = undo_prepocessing(target_sentences)

    save_data("eval/translations/greedy_translations", post_processed_sentences)

    return post_processed_sentences


def test_get_scores(model: nn.Module, source_data: List[List[str]], target_data: List[List[str]],
                    source_dict: Dictionary, target_dict: Dictionary, window_size: int):
    scores = get_scores(model, source_data, target_data, source_dict, target_dict, window_size)

    print(np.max(scores))
    print(len(scores))


def test_model_bleu(model: nn.Module, source_data: List[List[str]], reference_data: List[List[str]],
                    source_dict: Dictionary, target_dict: Dictionary, beam_size: int, window_size: int,
                    do_beam_search, translations: List[List[str]], use_torch_bleu=False):
    bleu_score = get_bleu_of_model(model, source_data, reference_data, source_dict, target_dict, beam_size, window_size,
                                   do_beam_search, translations, use_torch_bleu=use_torch_bleu)
    print(f"Model BLEU: {bleu_score}")


def determine_models_bleu(models_path: str, source_data: List[List[str]], reference_data: List[List[str]],
                          source_dict: Dictionary, target_dict: Dictionary, beam_size: int, window_size: int,
                          do_beam_search):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    directory = os.fsencode(models_path)

    scores = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pth"):
            model_path = os.path.join(models_path, filename)
            print(f"determine bleu of {model_path}")
            model = torch.load(model_path, map_location=device)

            bleu_score = get_bleu_of_model(model, source_data, reference_data, source_dict, target_dict, beam_size,
                                           window_size, do_beam_search, None)

            print(f"Model BLEU: {bleu_score}")
            scores.append((bleu_score, model_path))

    return scores
