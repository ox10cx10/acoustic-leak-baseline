"""Numerical and data-separation checks using synthetic waveforms."""

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from acoustic_leak import cross_validate, evaluate, extract_features, make_pipeline
from examples.synthetic import make_signals


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals, cls.labels, cls.groups = make_signals(n_groups=6, seed=21)

    def test_feature_schema_and_gain_invariance(self):
        signals = self.signals[:3]
        features, names = extract_features(signals)
        self.assertEqual(features.shape, (3, 93))
        self.assertEqual(len(set(names)), 93)
        self.assertTrue(np.isfinite(features).all())
        changed, changed_names = extract_features(signals * 4.0 + 3.0)
        self.assertEqual(names, changed_names)
        assert_allclose(features, changed, rtol=1e-9, atol=1e-9)
        quiet, _ = extract_features(signals * 1e-180)
        assert_allclose(features, quiet, rtol=1e-9, atol=1e-9)

    def test_invalid_recordings_are_rejected(self):
        invalid = [np.ones((1, 8192)), np.zeros((0, 8192)),
                   np.zeros((1, 8000)), self.signals[0],
                   np.full((1, 8192), np.nan), self.signals[:1].astype(complex)]
        for signals in invalid:
            with self.subTest(shape=signals.shape, dtype=signals.dtype):
                with self.assertRaises(ValueError):
                    extract_features(signals)
        with self.assertRaises(ValueError):
            extract_features(self.signals[:1], sample_rate=16000)

    def test_oof_predictions_use_only_the_training_fold(self):
        result = cross_validate(self.signals, self.labels, self.groups, C=0.1)
        scores, folds = result["scores"], result["folds"]
        self.assertEqual(scores.shape, self.labels.shape)
        self.assertEqual(len(np.unique(folds)), 3)
        for group in np.unique(self.groups):
            self.assertEqual(len(np.unique(folds[self.groups == group])), 1)
        for fold in np.unique(folds):
            valid = folds == fold
            model = make_pipeline(C=0.1).fit(self.signals[~valid], self.labels[~valid])
            expected = model.predict_proba(self.signals[valid])[:, 1]
            assert_allclose(scores[valid], expected, rtol=1e-9, atol=1e-9)
        self.assertEqual(result["metrics"], evaluate(self.labels, scores))

    def test_groups_and_labels_must_be_explicit(self):
        missing = self.groups.copy()
        missing[0] = ""
        for groups in (None, missing, self.groups[:-1]):
            with self.subTest(groups_type=type(groups).__name__):
                with self.assertRaises(ValueError):
                    cross_validate(self.signals, self.labels, groups)
        with self.assertRaises(ValueError):
            cross_validate(self.signals, np.zeros_like(self.labels), self.groups)

    def test_metrics_and_threshold(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9])
        metrics = evaluate(labels, scores)
        assert_array_equal(metrics["confusion_matrix"], [[2, 1], [1, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["leak_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["false_negative_rate"], 1 / 3)
        boundary = evaluate(np.array([0, 1]), np.array([0.5, 0.5]))
        assert_array_equal(boundary["confusion_matrix"], [[0, 1], [0, 1]])

    def test_saved_pipeline_keeps_feature_processing(self):
        model = make_pipeline(C=0.1).fit(self.signals, self.labels)
        expected = model.predict_proba(self.signals[:3])
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "model.joblib"
            joblib.dump(model, filename)
            restored = joblib.load(filename)
            assert_allclose(restored.predict_proba(self.signals[:3]), expected,
                            rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            model.predict_proba(self.signals[:1, :-1])


if __name__ == "__main__":
    unittest.main()
