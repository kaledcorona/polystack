import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from polystack import Polystack
from polystack.meta_features import MetaFeatures


# ---------------------------
# Single-view bypass (no stacking)
# ---------------------------
def test_single_view_bypass_predict_and_proba():
    X, y = make_classification(n_samples=180, n_features=8, n_informative=6, random_state=0)

    # Single estimator; Polystack should behave as a plain estimator
    base = LogisticRegression(max_iter=1000, random_state=0)
    ps = Polystack(estimators=[base], random_state=0)  # list is OK; becomes single view
    ps.fit(X, y)  # ndarray => wrapped as {"view1": X}

    # Should be in single-view mode and NOT have a final estimator
    assert ps.single_view_mode_ is True
    assert hasattr(ps, "single_estimator_")
    assert not hasattr(ps, "final_estimator_")

    # Predictions should match a directly-fitted model (same seed & data)
    ref = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    np.testing.assert_array_equal(ps.predict(X), ref.predict(X))
    # predict_proba is available on LR
    proba_ps = ps.predict_proba(X)
    assert proba_ps.shape == (X.shape[0], len(ps.classes_))


# ---------------------------
# Multi-view stacking (concat_proba)
# ---------------------------
def test_multiview_concat_proba_meta_and_predict():
    X, y = make_classification(
        n_samples=200, n_features=12, n_informative=8, n_redundant=0, n_classes=3, random_state=42
    )
    X_by_view = {
        "a": X[:, :6],
        "b": X[:, 6:],
    }
    ests = {
        "a": LogisticRegression(max_iter=1000, random_state=0),
        "b": RandomForestClassifier(n_estimators=60, random_state=0),
    }
    final = LogisticRegression(max_iter=1000, random_state=0)

    ps = Polystack(
        estimators=ests,
        final_estimator=final,
        cv=3,
        random_state=42,
        meta="concat_proba",   # fast baseline
    )
    ps.fit(X_by_view, y)

    assert ps.single_view_mode_ is False
    assert hasattr(ps, "final_estimator_")

    # meta features = per-view probs (C each) -> total V*C columns
    C = len(ps.classes_)
    assert len(ps.meta_feature_names_) == 2 * C

    y_pred = ps.predict(X_by_view)
    assert y_pred.shape == (X.shape[0],)

    # final estimator has predict_proba
    y_proba = ps.predict_proba(X_by_view)
    assert y_proba.shape == (X.shape[0], C)
    # rows should sum to ~1
    assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)


def test_stacking_is_deterministic_with_random_state():
    X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_classes=3,
    n_informative=3,
    random_state=0,
    )
    X_by_view = {"a": X[:, :5], "b": X[:, 5:]}

    kw = dict(
        estimators={"a": LogisticRegression(max_iter=500), "b": RandomForestClassifier(n_estimators=40)},
        final_estimator=LogisticRegression(max_iter=500),
        cv=3,
        random_state=42,
        meta="concat_proba",
    )

    p1 = Polystack(**kw).fit(X_by_view, y)
    p2 = Polystack(**kw).fit(X_by_view, y)

    # identical meta-feature names + identical preds
    assert p1.meta_feature_names_ == p2.meta_feature_names_
    np.testing.assert_array_equal(p1.predict(X_by_view), p2.predict(X_by_view))

# ---------------------------
# Estimator list mapping by view order
# ---------------------------
def test_estimators_list_maps_to_view_order():
    X, y = make_classification(n_samples=160, n_features=10, random_state=1)
    X_by_view = {"first": X[:, :5], "second": X[:, 5:]}

    # supply as list -> should map to ["first", "second"] in order
    ps = Polystack(
        estimators=[LogisticRegression(max_iter=500, random_state=0), RandomForestClassifier(n_estimators=40, random_state=0)],
        final_estimator=LogisticRegression(max_iter=500, random_state=0),
        cv=3,
        random_state=0,
        meta="concat_proba",
    )
    ps.fit(X_by_view, y)

    assert isinstance(ps.estimators_["first"], LogisticRegression)
    assert isinstance(ps.estimators_["second"], RandomForestClassifier)


# ---------------------------
# predict_proba error when final_estimator lacks it
# ---------------------------
def test_predict_proba_raises_if_final_estimator_has_no_proba():
    X, y = make_classification(n_samples=140, n_features=8, random_state=0)
    X_by_view = {"v1": X[:, :4], "v2": X[:, 4:]}

    # SVC(probability=False) has no predict_proba
    ps = Polystack(
        estimators={"v1": LogisticRegression(max_iter=500), "v2": RandomForestClassifier(n_estimators=50)},
        final_estimator=SVC(probability=False, random_state=0),
        cv=3,
        random_state=0,
        meta="concat_proba",
    )
    ps.fit(X_by_view, y)

    with pytest.raises(AttributeError):
        _ = ps.predict_proba(X_by_view)


# ---------------------------
# Wrong keys at predict -> error
# ---------------------------
def test_view_key_mismatch_on_predict_raises():
    X, y = make_classification(n_samples=120, n_features=8, random_state=0)
    X_by_view = {"a": X[:, :4], "b": X[:, 4:]}

    ps = Polystack(
        estimators={"a": LogisticRegression(max_iter=500), "b": RandomForestClassifier(n_estimators=30)},
        cv=3,
        random_state=0,
        meta="concat_proba",
    ).fit(X_by_view, y)

    with pytest.raises(ValueError):
        ps.predict({"a": X[:, :4], "c": X[:, 4:]})  # wrong key set


# ---------------------------
# _more_tags
# ---------------------------
def test_more_tags_values():
    ps = Polystack()
    tags = ps._more_tags()
    assert tags.get("requires_y") is True
    assert "no_validation" in tags
    assert "X_types" in tags  # we accept dicts but advertise conservative X_types


# ---------------------------
# Custom plugin instance
# ---------------------------
class TinyPlugin(MetaFeatures):
    def fit(self, preds_dict, probs_dict, classes, view_order) -> MetaFeatures:
        self.cols_ = ["tiny"]
        return self

    def transform(self, preds_dict, probs_dict, classes, view_order) -> pd.DataFrame:
        # Just one constant meta-feature for testing the hook
        n = next(iter(probs_dict.values())).shape[0]
        return pd.DataFrame({"tiny": np.ones(n)})


def test_custom_plugin_instance():
    X, y = make_classification(n_samples=100, n_features=6, n_classes=2, random_state=0)
    X_by_view = {"v1": X[:, :3], "v2": X[:, 3:]}

    ps = Polystack(
        estimators={"v1": LogisticRegression(max_iter=300), "v2": RandomForestClassifier(n_estimators=20)},
        final_estimator=LogisticRegression(max_iter=300),
        cv=3,
        random_state=0,
        meta=TinyPlugin(),   # pass instance directly
    )
    ps.fit(X_by_view, y)

    assert ps.single_view_mode_ is False
    assert ps.meta_feature_names_ == ["tiny"]
    yhat = ps.predict(X_by_view)
    assert yhat.shape == (X.shape[0],)

