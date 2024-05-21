from typing import List
from preprocessing.BPE import perform_bpe
import pickle


class Dictionary:
    # Initialize the dictionary
    def __init__(self, data: List[List[str]], operations: List[List[str]], transformed_words=None):
        self._word_to_idx_vocab = dict()
        self._idx_to_word_vocab = dict()
        self._operations = operations

        self._generate_vocabulary(data, operations, transformed_words)

    def __str__(self):
        # Return a string representation of the dictionary's contents
        return str(list(self._idx_to_word_vocab.items()))

    def __len__(self):
        # Return the size of the vocabulary
        return len(self._word_to_idx_vocab)

    def __contains__(self, item):
        # Check if a specific index or string is in the dictionary
        if isinstance(item, int):
            return item in self._idx_to_word_vocab
        elif isinstance(item, str):
            return item in self._word_to_idx_vocab

        return False

    def get_string_at_index(self, index: int):
        # Retrieve the string at a given index in the dictionary or raise a KeyError
        if index in self:
            return self._idx_to_word_vocab[index]

        return self._idx_to_word_vocab[0] # return UNK token

    def get_index_of_string(self, string: str):
        # Get the index of a given string in the dictionary
        if string in self:
            return self._word_to_idx_vocab[string]

        return 0  # If not found, return UNK

    def add_string(self, value: str):

        index = len(self)

        self._idx_to_word_vocab[index] = value
        self._word_to_idx_vocab[value] = index

    def empty(self):
        # Clear the dictionary's contents
        self._word_to_idx_vocab.clear()
        self._idx_to_word_vocab.clear()

    def _generate_vocabulary(self, data: List[List[str]], operations: List[List[str]], transformed_words=None):
        # save the words with operations applied to them so that the
        # operations don't have to be performed twice for the same word
        if not transformed_words:
            _, transformed_words = perform_bpe(data, operations)

        self.empty()
        self.add_string("<UNK>")
        self.add_string("<s>")
        self.add_string("</s>")
        for split_word in transformed_words.values():
            for token in split_word.split():
                if not (token in self):
                    self.add_string(token)

    def apply_vocabulary_to_text(self, data: List[List[str]], bpe_performed=False):
        if not bpe_performed:
            data, _ = perform_bpe(data, self._operations)

        new_data = []
        for sentence in data:
            new_sentence = []
            for token in sentence:
                if token in self:
                    new_sentence.append(token)
                else:
                    new_sentence.append(self.get_string_at_index(0))  # UNK
            new_data.append(new_sentence)
        return new_data

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

