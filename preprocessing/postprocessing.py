from typing import List
from preprocessing.BPE import undo_bpe


def delete_end_of_sentence_symbol(data: List[List[str]]):
    for sentence in data:
        sentence.pop()  # delete last symbol

    return data


def undo_prepocessing(data: List[List[str]]):
    # TODO replace UNK tokens with original tokens from source

    data = delete_end_of_sentence_symbol(data)

    data = undo_bpe(data)

    return data
