from metrics.metrics import WER, PER, BLEU
import argparse
from torchmetrics.text import WordErrorRate, BLEUScore
from typing import List, Union


def _load_data(path: str,
               split: bool = True) -> Union[List[str], List[List[str]]]:
    """
    Load data from a file.
    Args:
        path: The path to the file.
        split: Whether to split the data by whitespace.
    Returns: List of strings or list of lists of strings.

    """
    in_file = open(path, 'r', encoding='utf-8')

    if split:
        data = [line.split() for line in in_file]
    else:
        data = [line for line in in_file]
    in_file.close()

    return data


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-hy', '--hyps', type=str)
    parser.add_argument('-r', '--refs', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    hyps = _load_data(args.hyps)
    refs = _load_data(args.refs)

    # get the split data for torch metrics
    torch_hyps = _load_data(args.hyps, split=False)
    torch_refs = _load_data(args.refs, split=False)

    wer = WER()
    corpus_wer = wer(hyps, refs, on_corpus=True)
    print(f"WER on corpus: {corpus_wer}")

    torch_wer = WordErrorRate()
    print(f"Verify WER on corpus with torch: {torch_wer(torch_hyps, torch_refs)}")


    per = PER()
    corpus_per = per(hyps, refs, on_corpus=True)
    print(f"PER on corpus: {corpus_per}")


    bleu = BLEU()
    corpus_bleu = bleu(hyps, refs)
    print(f"BLEU on corpus: {corpus_bleu}")

    torch_bleu = BLEUScore()
    print(f"Verify BLEU on corpus with torch: {torch_bleu(torch_hyps, [[ref] for ref in torch_refs])}")

    print(wer.levenshtein(list("banane"), list("ananas"), verbose=True))
