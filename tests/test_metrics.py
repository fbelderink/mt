import unittest
from metrics.metrics import WER, PER, BLEU
from torchmetrics.text import WordErrorRate, BLEUScore, ExtendedEditDistance


class MetricsTest(unittest.TestCase):
    def test_wer(self):
        wer = WER()
        torch_wer = WordErrorRate()
        self.assertEqual(wer(["I", "am", "a", "test"], ["I", "am", "a", "test"]),
                         torch_wer(["I am a test"], ["I am a test"]))
        self.assertEqual(wer(["I", "am", "a", "test"], ["I", "am", "not", "a", "test"]),
                         torch_wer(["I am a test"], ["I am not a test"]))
        self.assertEqual(wer([], ["I", "am", "a", "test"]),
                         1)
        self.assertEqual(wer(["I", "am", "a", "test"], []),
                         1)


    def test_per(self):
        per = PER()
        self.assertEqual(per(["I", "am", "a", "test"], ["I", "am", "a", "test"]), 0.0)
        self.assertEqual(per(["I", "am", "a", "test"], ["I", "am", "not", "a", "test"]), 0.2)

    def test_levenshtein(self):
        torch_wer = WordErrorRate()
        wer = WER()
        self.assertEqual(wer.levenshtein(["You", "am", "a", "test"], ["I", "not", "a", "test", "something"]),
                         int(torch_wer(["You am a test"], ["I not a test something"]).item() * 5))
        self.assertEqual(wer.levenshtein([], ["I", "am", "a", "test"]),
                         4) # torch WER can't handle empty lists
        self.assertEqual(wer.levenshtein(["I", "am", "a", "test"], []),
                         4)
        self.assertEqual(wer.levenshtein(["I", "am", "a", "test"], ["I", "am", "a", "test"]),
                         int(torch_wer(["I am a test"], ["I am a test"]).item() * 4))
        self.assertEqual(wer.levenshtein(["I", "am", "a", "test"], ["I", "am", "not", "a", "test"]),
                         int(torch_wer(["I am a test"], ["I am not a test"]).item() * 5))

    def test_bleu(self):
        bleu = BLEU()
        torch_bleu = BLEUScore(n_gram=3)
        preds = ["I am you"]
        refs = [["I am you"]]
        torch_bleu.update(preds, refs)
        self.assertEqual(bleu([["I", "am", "you"]], [["I", "am", "you"]]),
                         torch_bleu.compute().item())

        preds = ["I am a test"]
        refs = [["I am a test"]]
        torch_bleu = BLEUScore(n_gram=4)
        torch_bleu.update(preds, refs)

        self.assertEqual(bleu([["I", "am", "a", "test"]], [["I", "am", "a", "test"]]),
                         torch_bleu.compute().item())

        torch_bleu = BLEUScore(n_gram=4)
        preds = ["I am a test"]
        refs = [["I am not a test"]]
        torch_bleu.update(preds, refs)

        self.assertEqual(bleu([["I", "am", "a", "test"]], [["I", "am", "not", "a", "test"]]),
                         torch_bleu.compute().item())


if __name__ == '__main__':
    unittest.main()
