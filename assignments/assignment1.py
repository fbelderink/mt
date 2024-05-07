from utils.file_manipulation import load_data
from metrics.metrics import WER, PER, BLEU
from torchmetrics.text import WordErrorRate, BLEUScore


def first_assignment(args, hyps, refs):
    # get the split data for torch metrics
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
