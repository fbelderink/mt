from typing import List


def _n_grams(input_str: List[str],
             n: int) -> List[List[str]]:
    """
    Generates n-grams from a list of strings.

    Parameters:
    input (List[str]): The input list of strings to generate n-grams from.
    n (int): The n-gram size.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0.")
    return [input_str[i:i + n] for i in range(len(input_str) - n + 1)]


def _n_gram_matches(source: List[str],
                    target: List[str],
                    n: int) -> int:
    """
    Evaluates the number of n-gram matches between the hypothesis and reference strings.

    Parameters:
    hyp (List[str]): The hypothesis string.
    ref (List[str]): The reference string.
    n (int): The n-gram size.

    Returns:
    int: The number of n-gram matches between the hypothesis and reference strings.
    """
    source_n_grams = _n_grams(source, n)
    target_n_grams = _n_grams(target, n)
    return sum(1 for n_gram in source_n_grams if n_gram in target_n_grams)


def bleu(hyp: List[str], 
         ref: List[str],
         n: int) -> float:
    ...


def _matches(hyp: List[str], 
             ref: List[str]) -> int:
    """
    Evaluates the number of matches between the hypothesis and reference strings.

    Parameters:
    hyp (List[str]): The hypothesis string.
    ref (List[str]): The reference string.

    Returns:
    int: The number of matches between the hypothesis and reference strings. 
    """
    return sum(1 for word in hyp if word in ref)


def per(hyp: List[str], 
        ref: List[str]) -> float: 
    """
    Evaluates the position-independent error rate (PER) between the hypothesis and reference strings.

    Parameters:
    hyp (List[str]): The hypothesis string.
    ref (List[str]): The reference string.

    Returns:
    float: The position-independent error rate (PER) between the hypothesis and reference strings. 
    """
    return 1 - (_matches(hyp, ref) - max(0, len(hyp) - len(ref)) / len(ref))


def wer(hyp: List[str], 
        ref: List[str]) -> float: 
    """
    Evaluates the Word Error Rate (WER) between the hypothesis and reference strings.

    Parameters:
    hyp (List[str]): The hypothesis string.
    ref (List[str]): The reference string.

    Returns:
    float: The Word Error Rate (WER) between the hypothesis and reference strings. 
    """
    if len(hyp) == 0:
        return len(ref) # as many insertions as there are words in the reference
    elif len(ref) == 0:
        return len(hyp) # as many deletions as there are words in the hypothesis
    return levenshtein_distance(hyp, ref) / len(ref)


def levenshtein_distance(hyp,
                         ref) -> int:
    """
    Evaluates the Levenshtein distance. 

    Parameters:
    hyp (str/List[str]): The hypothesis string / list of strings.
    ref (str/List[str]): The reference string / list of strings.

    Returns:
    int: The Levenshtein distance between the hypothesis and reference strings.
    """
    # cover edge cases
    if hyp == "":
        return len(ref)
    elif ref == "":
        return len(hyp)
    elif hyp == ref:
        return 0

    # length of strings
    K = len(hyp)
    L = len(ref)

    # initialization of arrays
    arr = [[-1 for _ in range(K + 1)] for _ in range(L + 1)]

    # initializing first row -> implicitly sets arr[0][0] = 0
    for i in range(K + 1):
        arr[0][i] = i

    # initializing first column
    for i in range(L + 1):
        arr[i][0] = i

    # running levenshtein
    for i in range(1, L + 1):
        for j in range(1, K + 1):
            if hyp[j - 1] == ref[i - 1]:
                arr[i][j] = arr[i - 1][j - 1]
            else:
                arr[i][j] = min(
                    arr[i - 1][j - 1] + 1, # substitution
                    arr[i][j - 1] + 1, # insertion
                    arr[i - 1][j] + 1 # deletion
                )
    return arr[L][K]


def main():
    # both strings empty
    assert levenshtein_distance("", "") == 0
    # one string empty
    assert levenshtein_distance("", "test") == 4
    assert levenshtein_distance("hello", "") == 5
    # both strings identical
    assert levenshtein_distance("identical", "identical") == 0
    # edit distance 1
    assert levenshtein_distance("test", "tent") == 1
    assert levenshtein_distance("cat", "cart") == 1
    assert levenshtein_distance("dart", "art") == 1
    # edit distance > 1
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("intention", "execution") == 5
    # longer strings
    assert levenshtein_distance("characteristically", "uncharacteristically") == 2
    assert levenshtein_distance("abcdefghijklmnopqrstuvwxyz", "zyxwvutsrqponmlkjihgfedcba") > 10
    # case sensitivity
    assert levenshtein_distance("Case", "case") == 1
    # different characters
    assert levenshtein_distance("hello world", "hello_world") == 1
    assert levenshtein_distance("function()", "function[]") == 2

    print(_matches(["bananas", "abc"], ["ananas",  "abc"]))


if __name__ == '__main__':
    main()
