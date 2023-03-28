"""No transformation, no fitting, no change."""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """IdentityTransformer. Nothing really happens here."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit."""
        self.coef_ = np.array([np.inf for _ in range(X.shape[1])])
        self.feature_importances_ = np.array([np.inf for _ in range(X.shape[1])])
        return self

    def transform(self, X, y=None):
        """Transform."""
        return X
