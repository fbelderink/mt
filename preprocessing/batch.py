from typing import List
from preprocessing.dictionary import Dictionary
import numpy as np


def _get_alignment(index: int, target_len: int, source_len: int) -> int:
    """
    Maps every target position to exactly one source position, considering following constraints:
    1. monotony: b_i \leq b_{i + 1}
    2. EOS to EOS: b_{I + 1} = J + 1, where I + 1 = target_len and J + 1 = source_len

    Args:
        index: index in the
        target_len:
        source_len:

    Returns: aligned (source) index for a given target index

    """
    #  align start of sentence
    if index == 0:
        return 0

    #  align end of sentence (second constraint)
    if index == target_len:
        return source_len

    #  equally distribute indices in remaining space
    dist_factor = (source_len - 1) / (target_len - 1)

    return int(round(index * dist_factor))


def _append_eos_to_sentences(data: List[List[str]]):
    """
    Appends end of sentence (EOS) symbol to every sentence.
    Args:
        data: List of sentences to append EOS to

    Returns: List of sentences with EOS symbol appended.

    """
    return [sentence + ['</s>'] for sentence in data]


def _get_target_labels(target_data: List[List[str]]):
    """
    Flattens input list and reshapes it to vector form
    Args:
        target_data: List of sentences to get label vector for

    Returns: numpy array of shape (N, 1) with target labels

    """
    #  flatten data and reshape to shape (N, 1)
    flattened_data = np.hstack(target_data).reshape(-1, 1)

    return flattened_data


def _get_padded_sentence(sentence: List[str], indices: tuple):
    """
    Pads sentence if needed with start of sentence (SOS) and end of sentence (EOS) according to indices.
    Args:
        sentence:
        indices: start and end indices of subsentence

    Returns: subsentence specified by indices, padded if needed

    """

    padded_sentence = []

    # pad with SOS if first index is below 0
    if indices[0] <= 0:
        padded_sentence = ['<s>'] * (abs(indices[0]) + 1)

    if indices[1] <= len(sentence):
        # append requested subsentence
        padded_sentence += sentence[max(0, indices[0] - 1): indices[1]]
    else:
        # append requested subsentence and pad with EOS if last index is above sentence length
        padded_sentence += sentence[max(0, indices[0] - 1):] + (['</s>'] * (indices[1] - len(sentence)))

    return padded_sentence


def _get_target_window_matrix(target_data: List[List[str]], window_size: int):
    """
    Calculates target window matrix for given target data and window size
    Args:
        target_data: List of target sentences
        window_size: context restriction window

    Returns: target window matrix with shape (N, window_size), N is determined by the len of target data

    """
    word_matrices = []
    for sentence in target_data:
        # create empty matrix to be filled with words
        word_matrix = np.empty((len(sentence), window_size), dtype=object)

        for idx, word in enumerate(sentence):
            # place subsentence into idx-th row of matrix, according to definition
            word_matrix[idx, :] = _get_padded_sentence(sentence, (idx + 1 - window_size, idx))

        word_matrices.append(word_matrix)

    # return vertical stacked matrix of word matrices list
    return np.vstack(word_matrices)


def _get_source_window_matrix(source_data: List[List[str]], target_data: List[List[str]], window_size: int):
    """
    Calculates source window matrix for given source and target data and window size

    Args:
        source_data: List of source sentences, where each sentence is a list of words
        target_data: List of target sentences, where each sentence is a list of words
        window_size: context restriction window

    Returns: source window matrix with shape (N, window_size), N is determined by the len of target data

    """
    word_matrices = []

    for source_sentence, target_sentence in zip(source_data, target_data):
        # create empty matrix to be filled with words
        word_matrix = np.empty((len(target_sentence), 2 * window_size + 1), dtype=object)
        for i in range(len(target_sentence)):
            # calculate alignment of index with given target and source sentence length
            b_i = _get_alignment(i, len(target_sentence), len(source_sentence))
            # get (padded) subsentence considering alignment and fill content into matrix row
            word_matrix[i, :] = _get_padded_sentence(source_sentence, (b_i - window_size, b_i + window_size))

        word_matrices.append(word_matrix)

    # return vertical stacked matrix of word matrices list
    return np.vstack(word_matrices)


def create_batch(source_data: List[List[str]], target_data: List[List[str]], window_size, batch_size) -> List[tuple]:
    """
    Creates list of batches consisting of source window matrix, target window matrix and labels
    with batch_size rows each encoded in tuples.

    Args:
        source_data: List of source sentences, where each sentence is a list of words
        target_data: List of target sentences, where each sentence is a list of words
        window_size: context restriction window
        batch_size: batch size (first dimension of matrices)

    Returns: list of batches, where each batch is a tuple

    """

    # add end of sentence symbols to sentences
    target_data_with_eos = _append_eos_to_sentences(target_data)

    # calculate matrices for complete data
    source_window_mat = _get_source_window_matrix(source_data, target_data_with_eos, window_size)
    target_labels = _get_target_labels(target_data_with_eos)
    target_window_mat = _get_target_window_matrix(target_data_with_eos, window_size)

    assert source_window_mat.shape[0] == target_window_mat.shape[0] == target_labels.shape[0]

    # len of matrices for complete data
    overall_len = source_window_mat.shape[0]

    # last batch has remainder size
    inv_remainder = batch_size - (overall_len % batch_size)

    batches = []
    for i in range(0, overall_len + inv_remainder, batch_size):
        end_idx = min(i + batch_size, overall_len)

        # split matrices for complete data into batch sized matrices
        batches.append((source_window_mat[i: end_idx, :],
                        target_window_mat[i: end_idx, :],
                        target_labels[i: end_idx]))

    return batches


def get_index_batches(batches: List[tuple], source_dict: Dictionary, target_dict: Dictionary):
    """
    Calculates index batches consisting of source window matrix, target window matrix and labels

    Args:
        batches: string batches to get indices for
        source_dict: Dictionary that holds the indices of words in source data
        target_dict: Dictionary that holds the indices of words in target data

    Returns: batches with string replaced by indices according to dictionaries

    """
    index_batches = []
    for batch in batches:
        # unpack batch tuple
        S, T, L = batch

        # vectorize get_index_of_string functions of dicts (functions are applied element wise on np arrays)
        get_source_idx = np.vectorize(source_dict.get_index_of_string)
        get_target_idx = np.vectorize(target_dict.get_index_of_string)

        # apply vectorized functions to batch arrays
        index_batches.append((get_source_idx(S), get_target_idx(T), get_target_idx(L)))

    return index_batches
