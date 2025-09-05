from __future__ import annotations

from typing import Callable, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# Public plugin interface
# ---------------------------------------------------------------------

class MetaFeatures:
    """Interface for meta-feature builders.

    A builder may keep state learned on OOF features (e.g., an encoder or PCA),
    so it has a fit/transform API.
    """

    def fit(
        self,
        preds_dict: dict[str, NDArray[Any]],
        probs_dict: dict[str, NDArray[np.floating]],
        classes: NDArray[Any],
        view_order: list[str],
    ) -> MetaFeatures:
        return self

    def transform(
        self,
        preds_dict: dict[str, NDArray[Any]],
        probs_dict: dict[str, NDArray[np.floating]],
        classes: NDArray[Any],
        view_order: list[str],
    ) -> pd.DataFrame:  # shape (n_samples, n_meta_features)
        raise NotImplementedError


_REGISTRY: dict[str, Callable[..., MetaFeatures]] = {}


def register(name: str) -> Callable[[Callable[..., MetaFeatures] | type], Callable[..., MetaFeatures] | type]:
    def _decorator(cls_or_factory):
        _REGISTRY[name] = cls_or_factory  # class or factory callable
        return cls_or_factory
    return _decorator


def create(name: str, **kwargs: Any) -> MetaFeatures:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown meta-features strategy '{name}'. Available: {sorted(_REGISTRY)}")
    factory = _REGISTRY[name]
    obj = factory(**kwargs) if callable(factory) else factory  # type: ignore[misc]
    if isinstance(obj, type):
        obj = obj(**kwargs)  # type: ignore[call-arg]
    return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _avg_proba(
    probs_dict: dict[str, NDArray[np.floating]], view_order: list[str]
) -> NDArray[np.floating]:
    stack = np.stack([probs_dict[v] for v in view_order], axis=0)  # (V, n, C)
    return stack.mean(axis=0)  # (n, C)


def _entropy(P: NDArray[np.floating], eps: float = 1e-9) -> NDArray[np.floating]:
    Psafe = np.clip(P, eps, 1.0)
    return (-Psafe * np.log(Psafe)).sum(axis=1, keepdims=True)


def _margin(P: NDArray[np.floating]) -> NDArray[np.floating]:
    if P.shape[1] == 1:
        return np.abs(P[:, [0]])
    top2 = np.partition(P, -2, axis=1)[:, -2:]
    return (top2[:, -1] - top2[:, -2])[:, None]


# ---------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------

@register("concat_proba")
class ConcatProba(MetaFeatures):
    """Concatenate per-view probability vectors.

    Columns: ``{view}_proba_class_{c}``
    """

    def transform(self, preds_dict, probs_dict, classes, view_order) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []
        for v in view_order:
            P = probs_dict[v]
            cols = [f"{v}_proba_class_{c}" for c in classes]
            blocks.append(pd.DataFrame(P, columns=cols))
        return pd.concat(blocks, axis=1)


@register("avg_proba_ohe")
class AvgProbaPlusOHE(MetaFeatures):
    """Average probability across views + one-hot of per-view predicted labels.

    Columns: ``avg_proba_class_{c}`` plus ``ohe_{view}_class_{c}`` for each view.
    """

    def __init__(self) -> None:
        self._enc: OneHotEncoder | None = None

    def fit(self, preds_dict, probs_dict, classes, view_order) -> MetaFeatures:
        # Fit encoder on classes once
        self._enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        classes_col = np.asarray(classes).reshape(-1, 1)
        # We want the encoder to learn the label set; give it all unique labels as a column
        self._enc.fit(classes_col)
        return self

    def transform(self, preds_dict, probs_dict, classes, view_order) -> pd.DataFrame:
        assert self._enc is not None, "AvgProbaPlusOHE must be fitted before transform."
        # Average probabilities across views
        avg = _avg_proba(probs_dict, view_order)
        avg_df = pd.DataFrame(avg, columns=[f"avg_proba_class_{c}" for c in classes])

        # One-hot encode the predicted labels per view
        ohe_blocks: list[pd.DataFrame] = []
        for v in view_order:
            yhat = np.asarray(preds_dict[v]).reshape(-1, 1)
            O = self._enc.transform(yhat)
            cols = [f"ohe_{v}_class_{c}" for c in self._enc.categories_[0].tolist()]
            ohe_blocks.append(pd.DataFrame(O, columns=cols))

        return pd.concat([avg_df] + ohe_blocks, axis=1)


@register("proba_margin_entropy")
class ProbaMarginEntropy(MetaFeatures):
    """Per view: probabilities + margin + entropy (compact, informative).

    Columns per view: ``{view}_proba_class_{c}``, ``{view}_margin``, ``{view}_entropy``.
    """

    def transform(self, preds_dict, probs_dict, classes, view_order) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []
        for v in view_order:
            P = probs_dict[v]
            cols = [f"{v}_proba_class_{c}" for c in classes]
            base = pd.DataFrame(P, columns=cols)
            base[f"{v}_margin"] = _margin(P)
            base[f"{v}_entropy"] = _entropy(P)
            blocks.append(base)
        return pd.concat(blocks, axis=1)


__all__ = ["MetaFeatures", "register", "create", "ConcatProba", "AvgProbaPlusOHE", "ProbaMarginEntropy"]

