from metrics.metrics import WER, PER, BLEU
import argparse
from torchmetrics.text import WordErrorRate, BLEUScore


def load_data(path, split=True):
    in_file = open(path, 'r', encoding='utf-8')

    if split:
        data = [line.split() for line in in_file]
    else:
        data = [line for line in in_file]
    in_file.close()

    return data


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-hy', '--hyps', type=str)
    parser.add_argument('-r', '--refs', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    hyps = load_data(args.hyps)
    refs = load_data(args.refs)

    torch_hyps = load_data(args.hyps, split=False)
    torch_refs = load_data(args.refs, split=False)

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
