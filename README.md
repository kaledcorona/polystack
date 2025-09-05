# Polystack

> **Multi‑view stacking for scikit‑learn, with fast single‑view fallback and typed API.**

Polystack lets you train **one estimator per data source (a *view*)** and then learn a
**final estimator** on top of the out‑of‑fold (OOF) predictions from each view. If you pass a
**single view**, Polystack **automatically bypasses stacking** and behaves like a plain
scikit‑learn estimator (useful for baselines and research ablations).

This project is inspired by and extends the ideas in
*multiviewstacking* (Garcia‑Ceja, 2018/2024). See *Citation* below.

---

## Highlights

* ✅ **Multi‑view stacking** with any scikit‑learn estimator per view
* ✅ **Single‑view fast path** (no meta‑learner; behaves like the base estimator)
* ✅ **Plugin meta‑features**: swap strategies or register your own
* ✅ **Deterministic** when you pass `random_state`
* ✅ **Typed package** (`py.typed` included) — great with **mypy/pyright**
* ✅ **Clean public API**: `Polystack` and `oof_by_view`
* ✅ Works with any sklearn CV **splitter** or a simple `"skf"/"kf"/"group"/"timeseries"` string

---

## Installation

Requirements:

* Python **≥ 3.10**
* numpy, pandas
* scikit‑learn **≥ 1.2**

```bash
# when published on PyPI
pip install polystack

# local dev (recommended while iterating)
pip install -e .
```

---

## Quick start

### 1) Single‑view (no stacking)

If you provide exactly one view, Polystack trains **just that estimator**.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from polystack import Polystack

X, y = make_classification(n_samples=400, n_features=12, random_state=0)

# Single view → bypass stacking automatically
clf = Polystack(estimators=[LogisticRegression(max_iter=1000)], random_state=0)
clf.fit({"view1": X}, y)
print(clf.predict({"view1": X})[:5])
```

### 2) Multi‑view stacking

Two views: first 6 features and last 6 features. We pick a different base model per view,
then a logistic regression as the final estimator.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from polystack import Polystack

X, y = make_classification(n_samples=600, n_features=12, n_informative=8, n_classes=3, random_state=42)
X_by_view = {"a": X[:, :6], "b": X[:, 6:]}

ps = Polystack(
    estimators={
        "a": LogisticRegression(max_iter=1000, random_state=0),
        "b": RandomForestClassifier(n_estimators=100, random_state=0),
    },
    final_estimator=LogisticRegression(max_iter=1000, random_state=0),
    cv=5,
    random_state=42,
    meta="concat_proba",  # see other strategies below
)
ps.fit(X_by_view, y)
proba = ps.predict_proba(X_by_view)  # shape: (n_samples, n_classes)
```

---

## Meta‑feature strategies (plugins)

Polystack builds the **meta‑feature matrix** from per‑view OOF predictions using a pluggable
strategy. Choose one by name, or provide your own implementation.

Built‑ins:

* `"concat_proba"` — concatenate per‑view probability vectors (fast, strong baseline)
* `"avg_proba_ohe"` — average probs across views **+** one‑hot predicted label per view
* `"proba_margin_entropy"` — per view: probs + confidence margin + entropy

### Custom plugin example

```python
# meta_features.py already exposes: MetaFeatures, register, create
from polystack.meta_features import MetaFeatures, register
import pandas as pd
import numpy as np

@register("my_custom")
class MyCustom(MetaFeatures):
    def fit(self, preds_dict, probs_dict, classes, view_order):
        self._cols = ["vote_count"]
        return self
    def transform(self, preds_dict, probs_dict, classes, view_order):
        # simple cross‑view agreement: #views predicting the argmax class
        votes = np.stack([probs_dict[v].argmax(1) for v in view_order], axis=1)
        vc = (votes == votes[:, [0]]).sum(1)
        return pd.DataFrame({"vote_count": vc})

# use it
ps = Polystack(..., meta="my_custom")
```

---

## API

### `Polystack`

```python
Polystack(
  estimators: dict[str, Estimator] | list[Estimator] | None = None,
  final_estimator: Estimator | None = None,
  cv: int | str | sklearn splitter | None = 5,
  random_state: int | None = None,
  n_jobs: int | None = None,
  meta: str | MetaFeatures = "avg_proba_ohe",
)
```

* **Single view** → trains only the base estimator; no meta‑features or final estimator
* `estimators` may be a **dict keyed by view**, a **list** (mapped in view order), or `None`
  (defaults to `RandomForestClassifier` per view)
* `cv` accepts integers, a **splitter instance** (`StratifiedKFold`, `KFold`, `GroupKFold`,
  `TimeSeriesSplit`, …), or convenient strings: `"skf"`, `"kf"`, `"group"`, `"timeseries"`
* `meta` selects the meta‑feature plugin by **name** or accepts a **MetaFeatures instance**

**Methods**: `fit`, `predict`, `predict_proba`

### `oof_by_view`

Low‑level helper that computes OOF predictions per view; useful for diagnostics and research.

```python
from polystack import oof_by_view

oof, y_aligned, folds = oof_by_view(
    estimators, X_by_view, y,
    cv=5, method="auto", random_state=0, return_fold_indices=True,
)
```

---

## Typing & mypy

This package ships a **`py.typed` marker** (PEP 561). Type checkers will use the inline
annotations automatically.

Example `mypy.ini` (optional):

```ini
[mypy]
python_version = 3.10
strict = True
warn_unused_ignores = True

[mypy-polystack.*]
# The library is typed; no special configuration required
ignore_missing_imports = False
```

If you package this project yourself, ensure `py.typed` is included in wheels/sdists. With
setuptools, in `pyproject.toml`:

```toml
[tool.setuptools.package-data]
polystack = ["py.typed"]
```

---

## Reproducibility & performance tips

* Pass `random_state` to make CV and models deterministic.
* In experiments, pick a **simple meta strategy** (e.g., `"concat_proba"`) for speed.
* Keep `X_by_view` as NumPy arrays (avoid per‑iteration DataFrame conversion) for tight loops.

---

## Contributing / Dev

```bash
# run tests
PYTHONPATH=src pytest -q

# style & lint (optional)
pip install ruff black mypy
ruff check --fix . && black . && mypy src
```

---

## Citation

If you use this project in academic work, please cite the original paper and package:

Garcia‑Ceja, Enrique, et al. *Multi‑view stacking for activity recognition with sound and
accelerometer data.* Information Fusion 40 (2018): 45–56.

```
Enrique Garcia-Ceja (2024). multiviewstacking: A python implementation of the Multi-View Stacking algorithm.
Python package https://github.com/enriquegit/multiviewstacking
```

**BibTeX**

```
@Manual{MVS,
  title  = {multiviewstacking: A python implementation of the Multi-View Stacking algorithm},
  author = {Enrique Garcia-Ceja},
  year   = {2024},
  note   = {Python package},
  url    = {https://github.com/enriquegit/multiviewstacking}
}
```

---

## License

This repository is distributed under the terms of the **MIT License** (see `LICENSE`).
