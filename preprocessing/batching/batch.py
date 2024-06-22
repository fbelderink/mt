from typing import List
from preprocessing.dictionary import Dictionary
from preprocessing.batching.fragment import fragment_data
import numpy as np


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

    source_window_mat, target_window_mat, target_labels = fragment_data(source_data, target_data, window_size)

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
