from typing import List
from preprocessing.dictionary import Dictionary, START_SYMBOL, END_SYMBOL, UNK_SYMBOL
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
    return [sentence + [END_SYMBOL] for sentence in data]


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
        padded_sentence = [START_SYMBOL] * (abs(indices[0]) + 1)

    if indices[1] <= len(sentence):
        # append requested subsentence
        padded_sentence += sentence[max(0, indices[0] - 1): indices[1]]
    else:
        # append requested subsentence and pad with EOS if last index is above sentence length
        padded_sentence += sentence[max(0, indices[0] - 1):] + ([END_SYMBOL] * (indices[1] - len(sentence)))

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
            word_matrix[i, :] = _get_padded_sentence(source_sentence, (b_i - window_size + 1, b_i + window_size + 1))

        word_matrices.append(word_matrix)

    # return vertical stacked matrix of word matrices list
    return np.vstack(word_matrices)


def create_source_window_matrix(source_data: List[str], source_dict: Dictionary, window_size: int, target_length: int):

    # calculate matrices for complete data
    source_window_mat = _get_source_window_matrix([source_data], [[UNK_SYMBOL] * target_length], window_size)

    get_source_idx = np.vectorize(source_dict.get_index_of_string)

    return get_source_idx(source_window_mat)


def fragment_data(source_data: List[List[str]], target_data: List[List[str]], window_size: int) -> tuple:
    # add end of sentence symbols to sentences
    target_data_with_eos = _append_eos_to_sentences(target_data)

    # calculate matrices for complete data
    source_window_mat = _get_source_window_matrix(source_data, target_data_with_eos, window_size)
    target_labels = _get_target_labels(target_data_with_eos)
    target_window_mat = _get_target_window_matrix(target_data_with_eos, window_size)

    return source_window_mat, target_window_mat, target_labels


def fragment_data_to_indices(source_data: List[List[str]],
                             target_data: List[List[str]],
                             window_size: int,
                             source_dict: Dictionary,
                             target_dict: Dictionary):
    S, T, L = fragment_data(source_data, target_data, window_size)

    get_source_idx = np.vectorize(source_dict.get_index_of_string)
    get_target_idx = np.vectorize(target_dict.get_index_of_string)

    return get_source_idx(S), get_target_idx(T), get_target_idx(L)
