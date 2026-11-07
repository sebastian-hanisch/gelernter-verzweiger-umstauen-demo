"""Das gelernte Modell: Trainingsdaten, logistische Regression, Determinismus, Trefferquote (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import gvl_algorithm as A
import gvl_model as M
from gvl_scenario import make_bay


def test_fit_logistic_recovers_a_known_model_on_synthetic_data():
    rng = np.random.RandomState(0)
    X = np.hstack([np.ones((4000, 1)), rng.normal(size=(4000, 3))])
    w_true = np.array([0.5, 2.0, -1.0, 0.0])
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
    w = M.fit_logistic(X, y)
    assert w == pytest.approx(w_true, abs=0.25) and M.accuracy(X, y, w) > 0.75


def test_fit_logistic_is_deterministic_and_the_penalty_shrinks_the_weights():
    rng = np.random.RandomState(1)
    X = np.hstack([np.ones((500, 1)), rng.normal(size=(500, 2))])
    y = (X[:, 1] + 0.3 * rng.normal(size=500) > 0).astype(float)
    assert np.array_equal(M.fit_logistic(X, y), M.fit_logistic(X, y))
    assert np.linalg.norm(M.fit_logistic(X, y, l2=50.0)) < np.linalg.norm(M.fit_logistic(X, y, l2=1e-2))


def test_accuracy_counts_the_right_side_of_one_half():
    X = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 2.0], [1.0, -2.0]])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    assert M.accuracy(X, y, np.array([0.0, 1.0])) == 0.5 and M.accuracy(X, y, np.array([0.0, -1.0])) == 0.5
    assert M.accuracy(X[:2], y[:2], np.array([0.0, 1.0])) == 1.0


def test_collected_data_has_the_right_shape_and_labels_match_the_independent_values():
    X, y = M.collect([make_bay(3, 3, 1000 + sd) for sd in range(4)], 5)
    assert X.shape[1] == len(A.FEATURE_NAMES) == 7 and X.shape[0] == len(y) > 10 and set(np.unique(y)) <= {0.0, 1.0}
    assert (X[:, 0] == 1.0).all() and 0.2 < y.mean() < 1.0                                         # Achsenabschnitt immer 1; es gibt optimale und nicht optimale Kinder
    X2, y2 = M.collect([make_bay(3, 3, 1000 + sd) for sd in range(4)], 5)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)                                         # deterministisch


def test_training_on_a_small_cell_is_deterministic_and_learns_the_safe_stack():
    w1, n1, share1, acc1 = M.train(((3, 3, 12),))
    w2, n2, share2, acc2 = M.train(((3, 3, 12),))
    assert np.array_equal(w1, w2) and (n1, share1, acc1) == (n2, share2, acc2)
    assert w1[1] > 0 and acc1 > 0.65                                                              # „sicherer Stapel“ zählt positiv; besser als Raten (Anteil optimaler Kinder liegt bei 0,7)


def test_first_child_optimal_counts_states_on_optimal_paths():
    bays = [make_bay(3, 3, sd) for sd in range(10)]
    hit, total = M.first_child_optimal(bays, 5, A.order_fit)
    hit_r, total_r = M.first_child_optimal(bays, 5, A.order_random_factory(0))
    assert total == total_r > 20 and hit <= total and hit_r <= total_r and hit >= hit_r              # die Regel ordnet mindestens so gut wie der Zufall
    h2, t2 = M.first_child_optimal(bays, 5, A.order_fit)
    assert (hit, total) == (h2, t2)
