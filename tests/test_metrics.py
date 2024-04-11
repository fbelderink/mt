import unittest
from metrics import metrics


class TestMetrics(unittest.TestCase):
    def test_n_grams(self):
        # expect a ValueError when n is 0
        with self.assertRaises(ValueError):
            metrics._n_grams(["I", "am", "a", "test"], 0)
        self.assertEqual(metrics._n_grams(["I", "am", "a", "test"], 1), [["I"], ["am"], ["a"], ["test"]])
        self.assertEqual(metrics._n_grams(["I", "am", "a", "test"], 2), [["I", "am"], ["am", "a"], ["a", "test"]])
        self.assertEqual(metrics._n_grams(["I", "am", "a", "test"], 3), [["I", "am", "a"], ["am", "a", "test"]])
        self.assertEqual(metrics._n_grams(["I", "am", "a", "test"], 4), [["I", "am", "a", "test"]])

    def test_n_gram_matches(self):
        self.assertEqual(metrics._n_gram_matches(["I", "am", "a", "test"], ["test", "a", "am", "I"], 1), 4)
        self.assertEqual(metrics._n_gram_matches(["I", "am", "a", "test"], ["test", "a", "am", "I"], 2), 0)
        self.assertEqual(metrics._n_gram_matches(["I", "am", "a", "test"], ["I", "am", "a", "test"], 2), 3)
        self.assertEqual(metrics._n_gram_matches(["I", "am", "a", "test"], ["I", "am", "not", "a", "test"], 2), 2)

    def test_matches(self):
        self.assertEqual(metrics._matches(["I", "am", "a", "test"], ["I", "am", "a", "test"]), 4)
        self.assertEqual(metrics._matches(["I", "am", "a", "test"], ["I", "am", "not", "a", "test"]), 4)

    def test_per(self):
        self.assertEqual(metrics.per(["I", "am", "a", "test"], ["I", "am", "a", "test"]), 0.0)
        self.assertEqual(metrics.per(["I", "am", "a", "test"], ["I", "am", "not", "a", "test"]), 0.25)

    def test_wer(self):
        self.assertEqual(metrics.wer(["I", "am", "a", "test"], ["I", "am", "a", "test"]), 0.0)
        self.assertEqual(metrics.wer(["I", "am", "a", "test"], ["I", "am", "not", "a", "test"]), 0.2)
        self.assertEqual(metrics.wer(["I", "am", "a", "test"], []), 4)
        self.assertEqual(metrics.wer([], ["I", "am", "a", "test"]), 4)

    def test_levenshtein_distance(self):
        self.assertEqual(metrics.levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(metrics.levenshtein_distance("intention", "execution"), 5)


if __name__ == '__main__':
    unittest.main()