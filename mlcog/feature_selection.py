"""Feature selection."""
from numpy import arange

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from .identity_transformer import IdentityTransformer


class FeatureSelectionUtils:
    """Feature selection techniques.
    """

    @staticmethod
    def selector_estimators(model_type):
        """
        Return base estimators for feature selection of type `model_type`.
        """
        estimators = []

        if model_type == "random_forest":
            estimators.append(RandomForestClassifier(class_weight='balanced', random_state=1))
            
        elif model_type == "elastic_net":
            for l1_ratio in arange(0, 1.2, 0.2):
                for c_param in [0.01, 0.1, 1, 10, 100]:
                    estimators.append(LogisticRegression(penalty='elasticnet', class_weight='balanced', random_state=1, solver='saga', max_iter=1000, l1_ratio=l1_ratio, C=c_param))
        else:
            estimators.append(IdentityTransformer())

        return estimators

    @staticmethod
    def selector_param_grid(model_type):
        """
        Return a parameters grid for the `selector` and the `scaler` steps.
        """
        param_grid = {
                        "selector__estimator": FeatureSelectionUtils.selector_estimators(model_type),
                        "selector__threshold": [0, 1e-5, "mean", "median", "1.5*mean"],
                        "scaler": [MaxAbsScaler(), IdentityTransformer()]
                    }

        return param_grid
