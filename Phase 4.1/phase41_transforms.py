from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CardinalityReducer(BaseEstimator, TransformerMixin):
    """Reduce categorical cardinality by mapping rare values to a shared bucket."""

    def __init__(self, columns: list[str], min_frequency: int = 12, max_categories: int = 70):
        self.columns = columns
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.allowed_: dict[str, set[str]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "CardinalityReducer":
        X = X.copy()
        self.allowed_ = {}

        for col in self.columns:
            if col not in X.columns:
                continue
            vc = X[col].astype(str).value_counts(dropna=False)
            vc = vc[vc >= self.min_frequency]
            keep = set(vc.head(self.max_categories).index.tolist())
            keep.update({"Unknown_Player", "OTHER_PLAYER"})
            self.allowed_[col] = keep

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, allowed in self.allowed_.items():
            if col not in X.columns:
                continue
            as_str = X[col].astype(str)
            X[col] = as_str.where(as_str.isin(allowed), "OTHER_PLAYER")
        return X
