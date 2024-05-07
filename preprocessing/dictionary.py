from typing import List
from preprocessing.BPE import perform_bpe


class Dictionary:
    # Initialize the dictionary
    def __init__(self):
        self.vocabulary = dict()

    def __str__(self):
        # Return a string representation of the dictionary's contents
        return str(list(self.vocabulary.items()))

    def __len__(self):
        # Return the size of the vocabulary
        return len(self.vocabulary)

    def __contains__(self, item):
        # Check if a specific index or string is in the dictionary
        if isinstance(item, int):
            return item in self.vocabulary
        elif isinstance(item, str):
            return item in self.vocabulary.values()
        else:
            raise TypeError("Invalid type. Expected int or str.")

    def get_string_at_index(self, index: int):
        # Retrieve the string at a given index in the dictionary or raise a KeyError
        if index in self.vocabulary:
            return self.vocabulary[index]
        else:
            raise KeyError(f"Index {index} not found in the vocabulary")

    def set_string_at_index(self, index: int, value: str):
        # Set or update the string at a specific index in the dictionary
        self.vocabulary[index] = value

    def get_index_of_string(self, string: str):
        # Get the index of a given string in the dictionary
        for key, val in self.vocabulary.items():
            if val == string:
                return key
        return 0  # If not found, return None

    def add_string(self, value: str):
        # Find the first available index to add the specified string
        index = 0
        # Loop to find the first available index
        while index in self.vocabulary:
            index += 1
        self.vocabulary[index] = value
        return index  # Return the index where the string was added

    def empty(self):
        # Clear the dictionary's contents
        self.vocabulary.clear()

    def generate_vocabulary(self, data: List[List[str]], operations: List[List[str]], transformed_words=None):
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

    def apply_vocabulary_to_text(self, data: List[List[str]]):
        new_data = []
        for sentence in data:
            new_sentence = []
            for word in sentence:
                if word in self:
                    new_sentence.append(word)
                else:
                    new_sentence.append('<UNK>')
            new_data.append(new_sentence)
        return new_data
