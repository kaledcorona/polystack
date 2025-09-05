from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array, check_is_fitted

from .crossval import oof_by_view, _predict_matrix, _is_classifier
from .meta_features import MetaFeatures, create as create_meta_builder


# ---------------------------
# Helper utilities (functions)
# ---------------------------

def _validate_y(y: ArrayLike) -> NDArray[Any]:
    """Return a 1D array of valid classification targets."""
    y_flat = check_array(y, ensure_2d=False, dtype=None).ravel()
    check_classification_targets(y_flat)
    return y_flat


def _normalize_views(
    X: Mapping[str, ArrayLike] | NDArray[Any] | pd.DataFrame,
    *,
    dtype: Any = np.float64,
    require_2d: bool = True,
) -> dict[str, NDArray[np.floating]]:
    """Normalize input into a dict {view_name: np.ndarray} with consistent rows."""
    if isinstance(X, pd.DataFrame):
        out = {"view1": X.to_numpy(copy=True, dtype=dtype)}
    elif isinstance(X, np.ndarray):
        out = {"view1": np.array(X, dtype=dtype, copy=True)}
    elif isinstance(X, Mapping):
        if not X:
            raise ValueError("X mapping is empty.")
        out = {
            k: (v.to_numpy(dtype=dtype, copy=True) if isinstance(v, pd.DataFrame)
                else np.array(v, dtype=dtype, copy=True))
            for k, v in X.items()
        }
    else:
        raise TypeError(f"Unsupported type for X: {type(X)}")

    # Validate shape
    if require_2d:
        bad = {k: a.shape for k, a in out.items() if a.ndim != 2}
        if bad:
            raise ValueError(f"All views must be 2D arrays; invalid shapes: {bad}")

    # Validate consistent n_samples
    n_rows = {k: a.shape[0] for k, a in out.items()}
    if len(set(n_rows.values())) != 1:
        raise ValueError(
            f"All views must share the same number of samples; got rows: {n_rows}"
        )

    return out


def _normalize_estimators(
    estimators: dict[str, BaseEstimator] | list[BaseEstimator] | None,
) -> dict[str, BaseEstimator]:
    """Normalize estimators into a dict.

    If a list is provided, it is mapped by position using temporal keys
    ``__pos_i``. Validation ensures all values are sklearn estimators.
    """
    if estimators is None:
        return {}

    if isinstance(estimators, dict):
        if not all(isinstance(v, BaseEstimator) for v in estimators.values()):
            raise TypeError("All values in estimators dict must be BaseEstimator instances.")
        return dict(estimators)

    if isinstance(estimators, list):
        if not all(isinstance(e, BaseEstimator) for e in estimators):
            raise TypeError("All items in estimators list must be BaseEstimator instances.")
        return {f"__pos_{i}": est for i, est in enumerate(estimators)}

    raise TypeError("estimators must be None, list[BaseEstimator], or dict[str, BaseEstimator]")


def _resolve_estimators(
    X_dict: dict[str, np.ndarray],
    estimators: dict[str, BaseEstimator],
    default_estimator: BaseEstimator,
) -> dict[str, BaseEstimator]:
    """Return a mapping {view_key: cloned_estimator}, one per view.

    Policy:
    - named dict → map by name; fill missing with defaults; error on extras
    - positional (``__pos_i`` keys) → map by *insertion order* of views
    - empty → default per view
    """
    view_keys = list(X_dict.keys())  # preserve user order

    if not estimators:
        return {k: clone(default_estimator) for k in view_keys}

    # positional list case previously normalized to __pos_i
    pos_keys = [k for k in estimators if k.startswith("__pos_")]
    if pos_keys:
        if len(pos_keys) != len(estimators):
            raise ValueError("Mixed positional and named estimators not allowed.")
        if len(pos_keys) != len(view_keys):
            raise ValueError(
                f"Number of estimators ({len(pos_keys)}) must equal number of views ({len(view_keys)})."
                " Pass a dict keyed by view names or adjust the list length."
            )
        est_ordered = [estimators[f"__pos_{i}"] for i in range(len(view_keys))]
        return {vk: clone(e) for vk, e in zip(view_keys, est_ordered)}

    # named dict case
    est_keys = set(estimators.keys())
    view_set = set(view_keys)
    extra = sorted(est_keys - view_set)
    if extra:
        raise ValueError(f"Estimators names not found in views: {extra}. Valid view names: {view_keys}")

    out = {k: clone(estimators[k]) for k in (est_keys & view_set)}
    # fill missing with defaults
    for k in (view_set - est_keys):
        out[k] = clone(default_estimator)
    # return in view order
    return {k: out[k] for k in view_keys}


# ---------------------------
# Estimator
# ---------------------------

class Polystack(BaseEstimator, ClassifierMixin):
    """
    Multi-view stacking classifier.

    Each view provides its own feature matrix (same rows across views). One base
    estimator is trained per view; their out-of-fold predictions are combined into
    meta-features for a final estimator. If **only one view** is provided, the
    model automatically **disables stacking** and behaves like a plain estimator
    (fit that single estimator and use it at predict time).
    """

    def __init__(
        self,
        estimators: dict[str, BaseEstimator] | list[BaseEstimator] | None = None,
        final_estimator: BaseEstimator | None = None,
        cv: int | str | Any | None = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        meta: str | MetaFeatures = "avg_proba_ohe",
    ) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = 5 if cv is None else cv
        self.random_state = random_state
        self.n_jobs = -1 if n_jobs is None else n_jobs
        self.meta = meta

        # runtime flags
        self.single_view_mode_: bool = False

    # ---------- internal validation ----------
    def _validate_inputs(
        self,
        X: Mapping[str, ArrayLike] | np.ndarray | pd.DataFrame,
        y: ArrayLike,
        *,
        dtype: Any = np.float64,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, BaseEstimator]]:
        X_norm = _normalize_views(X, dtype=dtype)
        y_norm = _validate_y(y)
        # choose default estimator lazily
        if not hasattr(self, "default_estimator") or self.default_estimator is None:
            self.default_estimator = RandomForestClassifier(random_state=self.random_state)
        ests_norm = _normalize_estimators(self.estimators)
        base_estimators = _resolve_estimators(X_norm, ests_norm, default_estimator=self.default_estimator)
        return X_norm, y_norm, base_estimators

    def _fit_single_view(
        self,
        X_dict: dict[str, np.ndarray],
        y: np.ndarray,
        estimators: dict[str, BaseEstimator] | None = None,
    ) -> Polystack:
        view_name, Xv = next(iter(X_dict.items()))
        if estimators and view_name in estimators:
            est = clone(estimators[view_name])
        else:
            if not hasattr(self, "default_estimator") or self.default_estimator is None:
                self.default_estimator = RandomForestClassifier(random_state=self.random_state)
            est = clone(self.default_estimator)
        est.fit(Xv, y)
        self.single_view_mode_ = True
        self.single_view_name_ = view_name
        self.single_estimator_ = est
        self.classes_ = unique_labels(y)
        self.n_features_in_ = {view_name: Xv.shape[1]}
        self.view_keys_ = [view_name]
        return self

    # ---------- sklearn API ----------
    def fit(self, X: Mapping[str, ArrayLike] | ArrayLike | pd.DataFrame, y: ArrayLike) -> Polystack:
        X_dict, y_arr, base_estimators = self._validate_inputs(X, y, dtype=np.float64)

        # single-view bypass (no stacking)
        if len(X_dict) == 1:
            return self._fit_single_view(X_dict, y_arr, estimators=base_estimators)

        # multi-view stacking path
        self.view_keys_ = list(X_dict.keys())
        self.n_samples_, self.n_views_ = y_arr.shape[0], len(self.view_keys_)
        if not _is_classifier(y_arr):
            raise ValueError("Polystack is a classifier; y must be a classification target.")
        self.classes_ = unique_labels(y_arr)

        # Compute OOF features per view (prefer probabilities via method="auto")
        oof_dict, y_oof, _ = oof_by_view(
            estimators=base_estimators,
            X_by_view=X_dict,
            y=y_arr,
            cv=self.cv,
            method="auto",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            return_fold_indices=False,
        )

        # Build probs/preds dicts from OOF
        probs_dict: dict[str, NDArray[np.floating]] = {}
        preds_dict: dict[str, NDArray[Any]] = {}
        for v in self.view_keys_:
            P = oof_dict[v]
            probs_dict[v] = P
            preds_dict[v] = self.classes_[P.argmax(axis=1)]

        # Resolve meta-feature builder
        self.meta_builder_: MetaFeatures = (
            create_meta_builder(self.meta) if isinstance(self.meta, str) else self.meta
        )
        self.meta_builder_.fit(preds_dict, probs_dict, self.classes_, self.view_keys_)
        meta_df = self.meta_builder_.transform(preds_dict, probs_dict, self.classes_, self.view_keys_)
        self.meta_feature_names_: list[str] = list(meta_df.columns)
        metaX = meta_df.to_numpy(dtype=float)

        # Fit final estimator (default RF)
        self.final_estimator_ = (
            self.final_estimator if self.final_estimator is not None
            else RandomForestClassifier(random_state=self.random_state)
        )
        self.final_estimator_.fit(metaX, y_oof)

        # Fit base estimators on full data for inference
        self.estimators_: dict[str, BaseEstimator] = {}
        for v in self.view_keys_:
            est = clone(base_estimators[v])
            if hasattr(est, "set_params") and hasattr(est, "get_params"):
                params = est.get_params(deep=False)
                if "random_state" in params:
                    try:
                        est.set_params(random_state=self.random_state)
                    except Exception:
                        pass
            est.fit(X_dict[v], y_arr)
            self.estimators_[v] = est

        # Cache schema
        self.n_features_in_: dict[str, int] = {v: X_dict[v].shape[1] for v in self.view_keys_}
        return self

    def _meta_features_for(self, X: dict[str, ArrayLike]) -> NDArray[np.floating]:
        check_is_fitted(self, ["classes_"])
        Xd = _normalize_views(X)
        if list(Xd.keys()) != self.view_keys_:
            raise ValueError(f"View keys mismatch. Expected {self.view_keys_}, got {list(Xd.keys())}.")
        for v in self.view_keys_:
            if Xd[v].shape[1] != self.n_features_in_[v]:
                raise ValueError(
                    f"View '{v}' has {Xd[v].shape[1]} features; expected {self.n_features_in_[v]}."
                )

        # Single-view fast-path
        if self.single_view_mode_:
            est = self.single_estimator_
            P = _predict_matrix(est, Xd[self.single_view_name_], method="auto", classes=self.classes_)
            return P  # unused by predict() path

        # Build per-view probs + predicted labels on full data
        probs_dict: dict[str, NDArray[np.floating]] = {}
        preds_dict: dict[str, NDArray[Any]] = {}
        for v in self.view_keys_:
            est = self.estimators_[v]
            Z = _predict_matrix(est, Xd[v], method="auto", classes=self.classes_)
            probs_dict[v] = Z
            preds_dict[v] = self.classes_[Z.argmax(axis=1)]
        meta_df = self.meta_builder_.transform(preds_dict, probs_dict, self.classes_, self.view_keys_)
        return meta_df.to_numpy(dtype=float)

    def predict(self, X: Mapping[str, ArrayLike] | ArrayLike | pd.DataFrame) -> NDArray[Any]:
        if isinstance(X, Mapping):
            if self.single_view_mode_:
                return self.single_estimator_.predict(next(iter(_normalize_views(X).values())))
            metaX = self._meta_features_for(X)
            return self.final_estimator_.predict(metaX)
        # non-dict: assume single-view matrix
        if self.single_view_mode_:
            return self.single_estimator_.predict(np.asarray(X))
        metaX = self._meta_features_for({"view1": X})
        return self.final_estimator_.predict(metaX)

    def predict_proba(self, X: Mapping[str, ArrayLike] | ArrayLike | pd.DataFrame) -> NDArray[np.floating]:
        if isinstance(X, Mapping):
            if self.single_view_mode_:
                return self.single_estimator_.predict_proba(next(iter(_normalize_views(X).values())))
            metaX = self._meta_features_for(X)
            if not hasattr(self.final_estimator_, "predict_proba"):
                raise AttributeError("final_estimator does not support predict_proba")
            return self.final_estimator_.predict_proba(metaX)
        # non-dict: assume single-view matrix
        if self.single_view_mode_:
            return self.single_estimator_.predict_proba(np.asarray(X))
        metaX = self._meta_features_for({"view1": X})
        if not hasattr(self.final_estimator_, "predict_proba"):
            raise AttributeError("final_estimator does not support predict_proba")
        return self.final_estimator_.predict_proba(metaX)

    # scikit-learn tags
    def _more_tags(self):
        return {
            "requires_y": True,
            "non_deterministic": self.random_state is None,
            "X_types": ["2darray"],  # conservative; we accept dicts and validate ourselves
            "allow_nan": False,
            "no_validation": True,
        }


__all__ = ["Polystack"]
