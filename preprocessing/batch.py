from typing import List
from preprocessing.dictionary import Dictionary
import numpy as np


def _get_alignment(index: int, target_len: int, source_len: int) -> int:
    #  align start of sentence
    if index == 0:
        return 0

    #  align end of sentence
    if index == target_len:
        return source_len

    #  equally distribute indices in remaining space
    dist_factor = (source_len - 1) / (target_len - 1)

    return int(round(index * dist_factor))


def _append_eos_to_sentences(data: List[List[str]]):
    #  append end of sentence symbol to every sentence
    return [sentence + ['</s>'] for sentence in data]


def _get_target_labels(target_data: List[List[str]]):
    #  flatten data and reshape to shape (N, 1)
    target_data = [[word + ('␇' if word != '</s>' else '') for word in sentence] for sentence in target_data]
    flattened_data = np.hstack(target_data).reshape(-1, 1)

    return flattened_data


def _get_padded_sentence(sentence: List[str], indices: tuple):
    padded_sentence = []
    if indices[0] <= 0:
        padded_sentence = ['<s>'] * (abs(indices[0]) + 1)

    if indices[1] <= len(sentence):
        padded_sentence += sentence[max(0, indices[0] - 1): indices[1]]
    else:
        padded_sentence += sentence[max(0, indices[0] - 1):] + (['</s>'] * (indices[1] - len(sentence)))

    padded_sentence = [s + ('␇' if s not in ['<s>', '</s>', '<UNK>'] else '') for s in padded_sentence]

    return padded_sentence


def _get_target_window_matrix(target_data: List[List[str]], window_size: int):
    word_matrices = []
    for sentence in target_data:
        word_matrix = np.empty((len(sentence), window_size), dtype=object)
        for idx, word in enumerate(sentence):
            word_matrix[idx, :] = _get_padded_sentence(sentence, (idx + 1 - window_size, idx))

        word_matrices.append(word_matrix)

    return np.vstack(word_matrices)


def _get_source_window_matrix(source_data: List[List[str]], target_data: List[List[str]], window_size: int):
    word_matrices = []
    for source_sentence, target_sentence in zip(source_data, target_data):
        word_matrix = np.empty((len(target_sentence), 2 * window_size + 1), dtype=object)
        for i in range(len(target_sentence)):
            b_i = _get_alignment(i, len(target_sentence), len(source_sentence))
            word_matrix[i, :] = _get_padded_sentence(source_sentence, (b_i - window_size, b_i + window_size))

        word_matrices.append(word_matrix)

    return np.vstack(word_matrices)


def create_batch(source_data: List[List[str]], target_data: List[List[str]], window_size, batch_size) -> List[tuple]:
    target_data_with_eos = _append_eos_to_sentences(target_data)

    source_window_mat = _get_source_window_matrix(source_data, target_data_with_eos, window_size)
    target_labels = _get_target_labels(target_data_with_eos)
    target_window_mat = _get_target_window_matrix(target_data_with_eos, window_size)

    assert source_window_mat.shape[0] == target_window_mat.shape[0] == target_labels.shape[0]

    overall_len = source_window_mat.shape[0]

    inv_remainder = 200 - (overall_len % batch_size)

    batches = []
    for i in range(0, overall_len + inv_remainder, batch_size):
        end_idx = min(i + batch_size, overall_len)
        batches.append((source_window_mat[i: end_idx, :],
                        target_window_mat[i: end_idx, :],
                        target_labels[i: end_idx]))

    return batches


def get_index_batches(batches: List[tuple], hyps_dict: Dictionary, refs_dict: Dictionary):
    index_batches = []
    for batch in batches:
        S, T, L = batch

        get_hyps_idx = np.vectorize(hyps_dict.getIndexOfString)
        get_refs_idx = np.vectorize(refs_dict.getIndexOfString)

        index_batches.append((get_hyps_idx(S), get_refs_idx(T), get_refs_idx(L)))

    return index_batches
