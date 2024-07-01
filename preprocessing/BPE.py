from typing import List


# adds spaces between letters and
# adds ␇ after the word to indicate the end of word 
def _split_and_add_eow(word: str) -> str:
    return ' '.join(word) + '␇'


# generates a dictionary of all token pairs in a given string and number of occurrences
def _get_token_pair_occurrences(word: str) -> dict[str, int]:
    token_pair_occurrences = dict()
    subWordList = word.split()
    for i in range(len(subWordList) - 1):
        current_join = subWordList[i] + ' ' + subWordList[i + 1]
        if current_join in token_pair_occurrences:
            token_pair_occurrences[current_join] += 1
        else:
            token_pair_occurrences[current_join] = 1

    return token_pair_occurrences


# executes one pulling-together-operation on a single word
def _execute_operation(word: str, operation: List[str]) -> str:
    res = ''
    subWordList = word.split()

    for i in range(len(subWordList) - 1):
        if subWordList[i] == operation[0] and subWordList[i + 1] == operation[1]:
            res += subWordList[i]
        else:
            res += subWordList[i] + ' '

    res += subWordList[-1]

    return res


def generate_bpe(training_data: List[List[str]],
                 operations_number: int) -> List[List[str]]:
    operations = []

    # step 1
    # create a dictionary with split words and number of occurrences
    word_occurrences = dict()
    for sentence in training_data:
        for word in sentence:
            split_word = _split_and_add_eow(word)

            if split_word in word_occurrences:
                word_occurrences[split_word] += 1
            else:
                word_occurrences[split_word] = 1

    for i in range(operations_number):
        # step 2
        # determine token pairs and count occurrences using dictionary and save them into token_pairs
        token_pairs = dict()
        for word in word_occurrences:
            # fetch all token pairs with number of occurrence for current word
            current_token_pair_dict = _get_token_pair_occurrences(word)

            # update token_pairs dictionary by transferring token pairs of current word
            for token_pair in current_token_pair_dict:
                if token_pair in token_pairs:
                    # add token pair's number of occurrence in current word multiplied by total number of occurrence of
                    # the word the token pair is from to total count of occurrence of the token pair
                    token_pairs[token_pair] += (
                            current_token_pair_dict[token_pair] * word_occurrences[word]
                    )
                else:
                    token_pairs[token_pair] = (
                            current_token_pair_dict[token_pair] * word_occurrences[word]
                    )

        if not token_pairs:
            # nothing left to pull together; all subwords of every word have been pulled together
            break

        # step 3
        # determine token pair with the highest number of occurrences and add to "operations" list
        # if there are two tokens with same number of occurrence, the one with higher lexicographic order is chosen
        next_operation = max(token_pairs, key=lambda k: (token_pairs[k], k))
        operations.append(next_operation.split())

        # step 4
        # execute most recent pulling-together operation for every word in "word_occurrences" dictionary
        # create new dictionary with updated keys
        updated_word_occurrences = dict()
        for word in word_occurrences:
            updated_word = _execute_operation(word, operations[-1])
            updated_word_occurrences[updated_word] = word_occurrences[word]
        # replace old dictionary with new one
        word_occurrences = updated_word_occurrences

    return operations


def perform_bpe(data: List[List[str]], operations: List[List[str]]):
    # save the words with operations applied to them so that the operations don't have to be performed
    # twice for the same word
    split_sentences = []
    transformed_words = dict()

    for sentence in data:
        new_sentence = []
        for word in sentence:
            if word in transformed_words:
                split_word = transformed_words[word]
            else:
                split_word = _split_and_add_eow(word)
                # apply all operations to word
                for operation in operations:
                    split_word = _execute_operation(split_word, operation)

                transformed_words[word] = split_word

            new_sentence.extend(split_word.split())

        split_sentences.append(new_sentence)

    return split_sentences, transformed_words


def undo_bpe(bpe_data: List[List[str]]):
    normal_data = []
    for sentence in bpe_data:
        line = ' '.join(sentence)
        line = line.replace(' ', '')
        line = line.replace('␇', ' ')
        normal_data.append(line.split())
    return normal_data
