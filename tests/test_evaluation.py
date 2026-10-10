import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from metrics.clustering import clustering_accuracy, spectral_clustering_metrics


class EvaluationProtocolTest(unittest.TestCase):
    def test_labels_do_not_select_partitions_and_metrics_stay_paired(self):
        rng = np.random.RandomState(21)
        x = rng.normal(size=(36, 5))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        affinity = np.abs(x @ x.T)
        labels = np.repeat(np.arange(3), 12)
        permuted = rng.permutation(labels)
        kwargs = dict(verbose=False, n_init=4, random_state=7, kmeans_n_init=2)
        original = spectral_clustering_metrics(affinity, 3, labels, **kwargs)
        changed = spectral_clustering_metrics(affinity, 3, permuted, **kwargs)
        np.testing.assert_array_equal(original[3], changed[3])
        for i, prediction in enumerate(original[3]):
            self.assertAlmostEqual(original[0][i], clustering_accuracy(labels, prediction))
            self.assertAlmostEqual(original[1][i], normalized_mutual_info_score(labels, prediction))
            self.assertAlmostEqual(original[2][i], adjusted_rand_score(labels, prediction))

    def test_evaluation_does_not_consume_global_numpy_rng(self):
        labels = np.repeat(np.arange(3), 8)
        affinity = (labels[:, None] == labels[None, :]).astype(float) + 0.02
        np.random.seed(91)
        before = np.random.get_state()
        spectral_clustering_metrics(affinity, 3, labels, verbose=False, n_init=2)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == '__main__':
    unittest.main()
