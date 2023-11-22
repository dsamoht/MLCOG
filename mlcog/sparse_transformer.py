from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix


class ConvertToSparse(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return csr_matrix(X)

class ConvertFromSparse(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()