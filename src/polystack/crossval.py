from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    BaseCrossValidator,
)

__all__ = [
    "oof_by_view",
    "_is_classifier",
    "_make_splitter",
    "_set_estimator_n_jobs",
    "_set_random_state",
    "_predict_matrix",
]


# ------------------------------
# Small utilities
# ------------------------------

def _is_classifier(y: ArrayLike) -> bool:
    """Heuristic: integer/bool/object targets → classification; floats → regression."""
    arr = np.asarray(y)
    return not np.issubdtype(arr.dtype, np.floating)


def _make_splitter(
    cv: int | str | BaseCrossValidator | None,
    *,
    y: ArrayLike,
    rng: np.random.RandomState | None = None,
    groups: ArrayLike | None = None,
    splits: int = 5,
) -> BaseCrossValidator:
    """Create a splitting strategy.

    Rules:
    - None/int: StratifiedKFold for classification, else KFold (n_splits=5 or int)
    - strings: "skf", "kf", "group", "timeseries"
    - passing an actual splitter instance returns it unchanged
    """
    # Pass-through only for real sklearn splitters, not strings
    if isinstance(cv, BaseCrossValidator):
        return cv

    shuffle = True

    if isinstance(cv, str):
        key = cv.lower()
        if key in {"skf", "stratified", "stratifiedkfold"}:
            return StratifiedKFold(n_splits=splits, shuffle=shuffle, random_state=rng)
        if key in {"kf", "kfold"}:
            return KFold(n_splits=splits, shuffle=shuffle, random_state=rng)
        if key in {"group", "groupkfold"}:
            if groups is None:
                raise ValueError("GroupKFold requires 'groups' argument.")
            return GroupKFold(n_splits=splits)
        if key in {"timeseries", "ts", "timeseriessplit"}:
            return TimeSeriesSplit(n_splits=splits)
        raise ValueError(f"Unknown cv string: {cv}")

    # None or int -> choose by task type
    if cv is None:
        n_splits = splits
    else:
        n_splits = int(cv)

    if _is_classifier(y):
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=rng)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=rng)


def _set_estimator_n_jobs(est: BaseEstimator, n_jobs: int | None) -> BaseEstimator:
    if n_jobs is None:
        return est
    try:
        params = est.get_params(deep=False)
    except Exception:
        return est
    if "n_jobs" in params:
        try:
            est.set_params(n_jobs=n_jobs)
        except Exception:
            pass
    return est


def _set_random_state(est: BaseEstimator, rng: np.random.RandomState | None) -> BaseEstimator:
    if rng is None:
        return est
    try:
        params = est.get_params(deep=False)
    except Exception:
        return est
    if "random_state" in params:
        try:
            seed = int(rng.randint(0, 2**31 - 1))
            est.set_params(random_state=seed)
        except Exception:
            pass
    return est


def _predict_matrix(
    est: Any,
    X: ArrayLike,
    *,
    method: Literal["auto", "predict_proba", "decision_function", "predict"] = "auto",
    classes: ArrayLike | None = None,
) -> NDArray[np.floating]:
    X = np.asarray(X)

    def _to_2d(a: ArrayLike) -> NDArray[np.floating]:
        A = np.asarray(a)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        return A.astype(float, copy=False)

    if method == "auto":
        if hasattr(est, "predict_proba"):
            method = "predict_proba"
        elif hasattr(est, "decision_function"):
            method = "decision_function"
        else:
            method = "predict"

    if method == "predict_proba":
        proba = est.predict_proba(X)
        P = _to_2d(proba)
        # Align to global class order if provided
        if classes is not None and hasattr(est, "classes_"):
            est_classes = np.asarray(est.classes_)
            global_classes = np.asarray(classes)
            indices = [int(np.where(est_classes == c)[0][0]) for c in global_classes]
            P = P[:, indices]
        return P

    if method == "decision_function":
        scores = est.decision_function(X)
        return _to_2d(scores)

    pred = est.predict(X)
    return _to_2d(pred)


# ------------------------------
# Main API
# ------------------------------

def oof_by_view(
    estimators: dict[str, BaseEstimator],
    X_by_view: dict[str, ArrayLike],
    y: ArrayLike,
    *,
    cv: int | str | BaseCrossValidator | None = 5,
    method: Literal["auto", "predict_proba", "decision_function", "predict"] = "auto",
    n_jobs: int | None = None,
    fold_jobs: int | None = None,
    random_state: int | None = None,
    groups: ArrayLike | None = None,
    return_fold_indices: bool = False,
    n_splits: int | None = None,
) -> tuple[
    dict[str, NDArray[np.floating]],
    NDArray[Any],
    list[tuple[NDArray[np.int64], NDArray[np.int64]]] | None,
]:
    """Compute out-of-fold predictions per view.

    Returns
    -------
    oof_dict : dict[str, 2D array]
        For each view, an (n_samples, n_outputs) matrix.
    y_oof : array
        The original target array (1D) to align with OOF rows.
    folds : list[(train_idx, test_idx)] | None
        Only when ``return_fold_indices=True``.
    """
    if not isinstance(estimators, dict):
        raise TypeError("estimators must be a dict[str, BaseEstimator]")
    if not isinstance(X_by_view, dict):
        raise TypeError("X_by_view must be a dict[str, array-like]")
    if not X_by_view:
        raise ValueError("X_by_view is empty")

    view_shapes = {k: np.asarray(v).shape for k, v in X_by_view.items()}
    n_rows = {k: s[0] for k, s in view_shapes.items()}
    if len(set(n_rows.values())) != 1:
        raise ValueError(f"All views must share the same number of samples; got {view_shapes}")

    y_arr = np.asarray(y)
    n = y_arr.shape[0]
    rng = np.random.RandomState(random_state) if random_state is not None else None

    splitter = _make_splitter(cv, y=y_arr, rng=rng, groups=groups, splits=(n_splits or 5))
    splits: list[tuple[np.ndarray, np.ndarray]] = list(
        splitter.split(next(iter(X_by_view.values())), y_arr, groups)
    )

    classes: NDArray[Any] | None = np.unique(y_arr) if _is_classifier(y_arr) else None

    oof_dict: dict[str, NDArray[np.floating]] = {}

    for (tr_idx, te_idx) in splits:
        for view_name, Xv in X_by_view.items():
            base = estimators[view_name]
            est = clone(base)
            est = _set_estimator_n_jobs(est, fold_jobs or n_jobs)
            est = _set_random_state(est, rng)

            Xtr, Xte = np.asarray(Xv)[tr_idx], np.asarray(Xv)[te_idx]
            ytr = y_arr[tr_idx]

            est.fit(Xtr, ytr)
            Zte = _predict_matrix(est, Xte, method=method, classes=classes)

            if view_name not in oof_dict:
                oof_dict[view_name] = np.empty((n, Zte.shape[1]), dtype=float)
            oof_dict[view_name][te_idx, :] = Zte

    folds = splits if return_fold_indices else None
    return oof_dict, y_arr, folds
