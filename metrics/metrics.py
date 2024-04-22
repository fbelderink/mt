from abc import abstractmethod
import numpy as np
import warnings
from typing import List, Union


class Metric:
    """
    Abstract parent class for metrics.
    """
    @abstractmethod
    def __call__(self,
                 hyp: List[str],
                 ref: List[str]):
        ...


class WER(Metric):
    """
    Word Error Rate (WER) metric.
    """
    def __call__(self,
                 hyp: List[str],
                 ref: List[str],
                 on_corpus: bool = False) -> float:
        """
        Evaluates the Word Error Rate (WER) between the hypothesis and reference strings.

        Parameters:
        hyp (list): The hypothesis string.
        ref (list): The reference string.
        on_corpus (bool): Whether to evaluate the WER on a corpus.

        Returns:
        float: The Word Error Rate (WER) between the hypothesis and reference strings.
        """
        if len(hyp) == 0 or len(ref) == 0:
            return 1

        if on_corpus:
            return sum([self.levenshtein(h, r) for (h, r) in zip(hyp, ref)]) / sum([len(r) for r in ref])

        return self.levenshtein(hyp, ref) / len(ref)

    def levenshtein(self,
                    hyp: List[str],
                    ref: List[str],
                    verbose: bool = False) -> int:
        """
        Evaluates the Levenshtein distance.

        Parameters:
        hyp (str/List[str]): The hypothesis string / list of strings.
        ref (str/List[str]): The reference string / list of strings.

        Returns:
        int: The Levenshtein distance between the hypothesis and reference strings.
        """

        # check base cases
        if len(hyp) == 0:
            return len(ref)

        if len(ref) == 0:
            return len(hyp)

        if hyp == ref:
            return 0

        # initialize matrix dimensions (hyp vertical, ref horizontal) <- therefore insertion and deletion are flipped
        # insertion: ←, deletion: ↑ (matrix is also transposed relative to matrix in slides because of that)
        J = len(hyp)
        K = len(ref)

        mat = [[-1 for _ in range(K + 1)] for _ in range(J + 1)]

        for j in range(J + 1):
            mat[j][0] = j

        for k in range(K + 1):
            mat[0][k] = k

        # traverse matrix from left to right, top to bottom and apply levenshteins' recursion formula
        for j in range(1, J + 1):
            for k in range(1, K + 1):
                if hyp[j - 1] == ref[k - 1]:
                    mat[j][k] = mat[j - 1][k - 1]  # match
                else:
                    mat[j][k] = 1 + min(mat[j - 1][k - 1],  # substitution
                                        mat[j][k - 1],  # insertion
                                        mat[j - 1][k])  # deletion

        # print levenshtein edits if verbose mode is activated
        if verbose:
            traceback = self._traceback_levenshtein(mat, J, K)
            self._print_levenshtein_edits(hyp, ref, traceback)

        return mat[J][K]

    def _print_levenshtein_edits(self,
                                 hyp: List[str],
                                 ref: List[str],
                                 traceback: str) -> None:
        """
        Print the Levenshtein edits.

        Parameters:
        hyp (list): The hypothesis string.
        ref (list): The reference string.
        traceback (list): The traceback of the Levenshtein matrix.

        Returns:
        None
        """
        idx_ref = idx_hyp = 0
        for c in traceback:
            match c:
                case 'd':
                    print(f"delete {hyp[idx_hyp]}")
                    idx_hyp += 1
                case 's':
                    print(f"substitute {hyp[idx_hyp]} with {ref[idx_ref]}")
                    idx_ref += 1
                    idx_hyp += 1
                case 'm':
                    print(f"{hyp[idx_hyp]} and {ref[idx_ref]} match")
                    idx_ref += 1
                    idx_hyp += 1
                case 'i':
                    print(f"insert {ref[idx_ref]}")
                    idx_ref += 1

    def _traceback_levenshtein(self,
                               mat: List[List[int]],
                               j: int,
                               k: int) -> str:
        """
        Traceback the Levenshtein matrix to get the edit operations.

        Parameters:
        mat (list): The Levenshtein matrix.
        j (int): The current vertical position.
        k (int): The current horizontal position.

        Returns:
        str: The edit operations.
        """
        if j == 0:
            return "i" * k  # if k is not 0, we have k insertion steps (←) to get to (0, 0)

        if k == 0:
            return "d" * j  # if j is not 0, we have j deletion steps (↑) to get to (0, 0)

        possible = [mat[j - 1][k - 1], mat[j - 1][k], mat[j][k - 1]]  # possible predecessors in matrix

        # append taken action for each step and calc other steps recursively
        match possible.index(min(possible)):  # TODO: find out why we need to take the minimum
            case 0:  # match or substitution
                if mat[j][k] == mat[j - 1][k - 1]:
                    return self._traceback_levenshtein(mat, j - 1, k - 1) + "m"
                return self._traceback_levenshtein(mat, j - 1, k - 1) + "s"
            case 1:  # deletion
                return self._traceback_levenshtein(mat, j - 1, k) + "d"
            case 2:  # insertion
                return self._traceback_levenshtein(mat, j, k - 1) + "i"


class PER(Metric):
    """
    Position-independent error rate (PER) metric.
    """
    def __call__(self,
                 hyp: List[str],
                 ref: List[str],
                 on_corpus: bool = False) -> float:
        """
        Evaluates the position-independent error rate (PER) between the hypothesis and reference strings.

        Parameters:
        hyp (list): The hypothesis tokens.
        ref (list): The reference tokens.

        Returns:
        float: The position-independent error rate (PER) between the hypothesis and reference strings.
        """
        if on_corpus:
            h_l = sum([len(h) for h in hyp])
            r_l = sum([len(r) for r in ref])
            return 1 - ((sum([self._matches(h, r) for (h, r) in zip(hyp, ref)]) - max(0, h_l - r_l)) / r_l)

        return 1 - ((self._matches(hyp, ref) - max(0, len(hyp) - len(ref))) / len(ref))

    def _matches(self,
                 hyp: List[str],
                 ref: List[str]) -> int:
        """
        Evaluates the number of matches between the hypothesis and reference strings.

        Parameters:
        hyp (list): The hypothesis tokens.
        ref (list): The reference tokens.

        Returns:
        int: The number of matches between the hypothesis and reference strings.
        """
        return len([w for w in hyp if w in ref])


class BLEU(Metric):
    """
    Bilingual Evaluation Understudy (BLEU) metric.
    """
    def __call__(self,
                 hyps: List[List[str]],
                 refs: List[List[str]],
                 N: int = 4) -> float:
        """
        Evaluates the Bilingual Evaluation Understudy (BLEU) between the hypothesis and reference strings.

        Parameters:
        hyps (list): The hypothesis strings.
        refs (list): The reference strings.
        N (int): The maximum n-gram size.

        Returns:
        float: The BLEU Score
        """
        hyp_max_len = max([len(h) for h in hyps])
        ref_max_len = max([len(r) for r in refs])
        if N > hyp_max_len:
            N = hyp_max_len
        elif N > ref_max_len:
            N = ref_max_len
        # calculate modified n-gram precision up to N
        n_gram_precisions = [self._modified_n_gram_precision(hyps, refs, n) for n in range(1, N + 1)]

        modified_precision_sum = 0
        for n, prec in enumerate(n_gram_precisions):
            if prec == 0:
                warnings.warn(
                    f"The BLEU score evaluates to zero, because the modified {n}-gram precision evaluates to zero")
                return 0
            modified_precision_sum += np.log(prec)
        # calculate brevity penalty
        bp = self._brevity_penalty(hyps, refs)

        return bp * np.exp((1 / N) * modified_precision_sum)

    def _modified_n_gram_precision(self,
                                   hyps: List[List[str]],
                                   refs: List[List[str]],
                                   n: int) -> float:
        """
        Calculates the modified n-gram precision of given hypotheses, references and a specified n..
        Extends the unigram precision by considering longer sequences of words.

        Parameters:
        hyps (list): The hypothesis strings.
        refs (list): The reference strings.

        Returns:
        float: The modified n-gram precision.
        """
        numerator, denominator = 0, 0

        for (hyp, ref) in zip(hyps, refs):  # for all L (hyp, ref) pairs
            n_grams_hyp, n_grams_ref = self._n_grams(hyp, n), self._n_grams(ref, n)

            h_l = [list(x) for x in set(tuple(x) for x in n_grams_hyp)]  # get unique values only

            # count occurrences of current n_gram in ref and hyp
            numerator += sum([min(n_grams_hyp.count(n_gram), n_grams_ref.count(n_gram)) for n_gram in h_l])
            denominator += sum([n_grams_hyp.count(n_gram) for n_gram in h_l])

        return numerator / denominator

    def _brevity_penalty(self,
                         hyp: Union[List[str], List[List[str]]],
                         ref: Union[List[str], List[List[str]]]) -> float:
        """
        Calculates the brevity penalty for given hypotheses and references.
        The brevity penalty is relevant for hypotheses that are too short and which potentially allows them to have a perfect precision.

        Parameters:
        hyp (list): The hypothesis strings.
        ref (list): The reference strings.

        Returns:
        float: The brevity penalty.
        """
        # calculate accumulated length of hypotheses and references
        l_h, l_r = sum([len(h) for h in hyp]), sum([len(r) for r in ref])

        if l_h > l_r:
            return 1

        return np.exp(1 - (l_r / l_h))

    def _n_grams(self,
                 tokens: List[str],
                 n: int) -> List[List[str]]:
        """
        Generates n-grams from a list of tokens.

        Parameters:
        input (list): The input list of tokens to generate n-grams from.
        n (int): The n-gram size.
        """
        n_grams = []
        for i in range(len(tokens)):
            if i + n <= len(tokens):
                n_gram = [tokens[j] for j in range(i, i + n)]
                n_grams.append(n_gram)

        return n_grams
