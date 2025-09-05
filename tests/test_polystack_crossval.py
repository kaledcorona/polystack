import math
from typing import Any

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit

from polystack import crossval as pcv


# --------------------
# Helper dummy estimators
# --------------------
class NoJobsEstimator:
    """A minimal estimator without n_jobs/random_state params."""

    def fit(self, X, y):
        self.coef_ = np.mean(X, axis=0, keepdims=True)
        return self

    def predict(self, X):
        # simple linear score
        s = X @ self.coef_.T
        return np.ravel(s)


class FakeProbaEstimator:
    """
    Minimal object exposing predict_proba + classes_.
    Used only to unit test _predict_matrix alignment; no fitting.
    """

    def __init__(self, classes, proba):
        self.classes_ = np.asarray(classes)
        self._proba = np.asarray(proba)

    def predict_proba(self, X):  # X ignored on purpose
        n = len(X)
        # repeat the provided probability rows to length n
        if self._proba.shape[0] == 1 and n > 1:
            return np.repeat(self._proba, n, axis=0)
        return self._proba


class FakeDecisionEstimator:
    def __init__(self, out):
        self._out = np.asarray(out)

    def decision_function(self, X):
        n = len(X)
        if self._out.ndim == 1 and self._out.size == 1:
            return np.repeat(self._out, n)
        if self._out.ndim == 2 and self._out.shape[0] == 1:
            return np.repeat(self._out, n, axis=0)
        return self._out


# --------------------
# Unit tests for helpers
# --------------------

def test__is_classifier_true_false():
    y_cls = np.array([0, 1, 1, 0, 1])
    y_reg = np.array([0.1, 0.2, 0.3, 0.0])
    assert pcv._is_classifier(y_cls) is True
    assert pcv._is_classifier(y_reg) is False


def test__make_splitter_defaults_and_strings():
    rng = np.random.RandomState(0)
    y_cls = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    # Use float targets to signal regression to _is_classifier
    y_reg = np.linspace(0.0, 1.0, 8)

    # None -> stratified for classification, kfold for regression
    sp1 = pcv._make_splitter(None, y=y_cls, rng=rng)
    assert isinstance(sp1, StratifiedKFold)
    assert sp1.n_splits == 5

    sp2 = pcv._make_splitter(None, y=y_reg, rng=rng)
    assert isinstance(sp2, KFold)
    assert sp2.n_splits == 5

    # int -> same rule with that many splits
    sp3 = pcv._make_splitter(3, y=y_cls, rng=rng)
    assert isinstance(sp3, StratifiedKFold) and sp3.n_splits == 3
    sp4 = pcv._make_splitter(4, y=y_reg, rng=rng)
    assert isinstance(sp4, KFold) and sp4.n_splits == 4

    # strings
    sp5 = pcv._make_splitter("skf", y=y_cls, rng=rng)
    assert isinstance(sp5, StratifiedKFold)
    sp6 = pcv._make_splitter("kf", y=y_reg, rng=rng)
    assert isinstance(sp6, KFold)

    # group requires groups
    with pytest.raises(ValueError):
        pcv._make_splitter("group", y=y_cls, rng=rng)

    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    sp7 = pcv._make_splitter("group", y=y_cls, rng=rng, groups=groups)
    assert isinstance(sp7, GroupKFold) and sp7.n_splits == 5

    sp8 = pcv._make_splitter("timeseries", y=y_reg, rng=rng)
    assert isinstance(sp8, TimeSeriesSplit)

    with pytest.raises(ValueError):
        pcv._make_splitter("does-not-exist", y=y_cls, rng=rng)

    # Pass-through
    sp9 = StratifiedKFold(n_splits=2)
    assert pcv._make_splitter(sp9, y=y_cls, rng=rng) is sp9


def test__set_estimator_n_jobs_and_random_state():
    # n_jobs present
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rf2 = pcv._set_estimator_n_jobs(rf, 1)
    assert rf2.get_params()["n_jobs"] == 1

    # n_jobs missing -> no error, unchanged
    no = NoJobsEstimator()
    no2 = pcv._set_estimator_n_jobs(no, 2)
    assert isinstance(no2, NoJobsEstimator)
    assert not hasattr(no2, "n_jobs")

    # random_state present
    lr = LogisticRegression(max_iter=200)
    rs = np.random.RandomState(123)
    lr2 = pcv._set_random_state(lr, rs)
    assert isinstance(lr2.get_params()["random_state"], (int, np.integer))

    # random_state missing -> unchanged
    no3 = pcv._set_random_state(no, rs)
    assert isinstance(no3, NoJobsEstimator)


def test__predict_matrix_alignment_and_paths():
    X = np.zeros((5, 2))

    # predict_proba alignment: estimator says classes_ = [1, 0], global = [0, 1]
    fake = FakeProbaEstimator(classes=[1, 0], proba=np.array([[0.7, 0.3]]))
    out = pcv._predict_matrix(fake, X, method="auto", classes=np.array([0, 1]))
    # After alignment, column 0 must correspond to class 0 -> 0.3
    assert out.shape == (5, 2)
    assert np.allclose(out[0], [0.3, 0.7])

    # decision_function path with 1D output
    dec = FakeDecisionEstimator(out=np.array([1.0]))
    out2 = pcv._predict_matrix(dec, X, method="decision_function")
    assert out2.shape == (5, 1)

    # fallback to predict for regressors (2D ensured)
    Xr, yr = make_regression(n_samples=30, n_features=3, random_state=0)
    reg = LinearRegression().fit(Xr, yr)
    out3 = pcv._predict_matrix(reg, Xr, method="auto")
    assert out3.ndim == 2 and out3.shape[1] == 1


# --------------------
# Integration tests for oof_by_view
# --------------------

def test_oof_by_view_regression_shapes_and_y():
    Xr, yr = make_regression(n_samples=60, n_features=6, noise=0.1, random_state=0)
    # Two views: split the features
    X_by_view = {
        "a": Xr[:, :3],
        "b": Xr[:, 3:],
    }
    estimators = {
        "a": Ridge(alpha=1.0, random_state=0),
        "b": LinearRegression(),
    }

    oof, y_oof, folds = pcv.oof_by_view(
        estimators,
        X_by_view,
        yr,
        cv=4,
        method="predict",
        random_state=42,
        return_fold_indices=True,
    )

    assert set(oof.keys()) == {"a", "b"}
    n = len(yr)
    assert oof["a"].shape == (n, 1)
    assert oof["b"].shape == (n, 1)
    # y_oof preserves the original order/content
    assert np.array_equal(np.asarray(yr), y_oof)

    # Folds cover all indices exactly once in test sets
    test_cover = np.unique(np.concatenate([te for _, te in folds]))
    assert np.array_equal(test_cover, np.arange(n))


def test_oof_by_view_classification_proba_and_shapes():
    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        random_state=7,
    )
    X_by_view = {
        "x1": X[:, :5],
        "x2": X[:, 5:],
    }
    estimators = {
        "x1": LogisticRegression(max_iter=1000, random_state=0),
        "x2": RandomForestClassifier(n_estimators=40, random_state=0),
    }

    oof, y_oof, _ = pcv.oof_by_view(
        estimators,
        X_by_view,
        y,
        cv=3,
        method="auto",
        random_state=123,
    )

    n = len(y)
    for v in ("x1", "x2"):
        assert oof[v].shape == (n, 3)
        # Rows from predict_proba should sum (approximately) to 1
        row_sums = oof[v].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.array_equal(y_oof, np.asarray(y))


import importlib.util
HAS_JOBLIB = importlib.util.find_spec("joblib") is not None


@pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not available")
def test_oof_by_view_parallel_and_folds():
    X, y = make_classification(n_samples=90, n_features=6, random_state=0)
    X_by_view = {"v": X}
    estimators = {"v": LogisticRegression(max_iter=500)}
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    oof, y_oof, folds = pcv.oof_by_view(
        estimators,
        X_by_view,
        y,
        cv=cv,
        method="auto",
        random_state=0,
        n_jobs=2,  # trigger Parallel branch
        return_fold_indices=True,
    )

    assert folds is not None and len(folds) == 4
    # Ensure disjoint test indices covering all samples
    test_cover = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(test_cover), np.arange(len(y)))


# --------------------
# Error handling
# --------------------

def test_oof_by_view_input_validation_and_errors():
    X, y = make_classification(n_samples=40, n_features=5, random_state=0)
    estimators = {"only": LogisticRegression(max_iter=200)}

    # estimators and X_by_view must be dicts
    with pytest.raises(TypeError):
        pcv.oof_by_view([], {"only": X}, y)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pcv.oof_by_view(estimators, X, y)  # type: ignore[arg-type]

    # missing view
    with pytest.raises(ValueError):
        pcv.oof_by_view(estimators, {}, y)

    # mismatched rows across views
    with pytest.raises(ValueError):
        pcv.oof_by_view(estimators, {"only": X[:-1]}, y)

    # unknown cv string
    with pytest.raises(ValueError):
        pcv.oof_by_view(estimators, {"only": X}, y, cv="nope")

    # group splitter without groups
    with pytest.raises(ValueError):
        pcv.oof_by_view(estimators, {"only": X}, y, cv="group")


# --------------------
# Determinism
# --------------------

def test_oof_by_view_deterministic_with_random_state():
    X, y = make_classification(n_samples=70, n_features=8, random_state=0)
    X_by_view = {"v1": X[:, :4], "v2": X[:, 4:]}
    ests = {"v1": LogisticRegression(max_iter=300), "v2": RandomForestClassifier(n_estimators=25)}

    out1 = pcv.oof_by_view(ests, X_by_view, y, cv=3, random_state=42)
    out2 = pcv.oof_by_view(ests, X_by_view, y, cv=3, random_state=42)

    oof1, y1, _ = out1
    oof2, y2, _ = out2

    assert np.array_equal(y1, y2)
    for v in oof1:
        assert np.allclose(oof1[v], oof2[v])


# --------------------
# Public API
# --------------------

def test_public_api_contains_oof_by_view():
    import polystack as pkg
    assert hasattr(pkg, "oof_by_view")
    if hasattr(pkg, "__all__"):
        assert "oof_by_view" in pkg.__all__
