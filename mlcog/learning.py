"""Main learning."""
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, make_scorer)

from .estimators import Estimators
from .feature_selection import FeatureSelectionUtils


class Learning:
    """
    Main learning.
    """
    
    RANDOM_STATE = 0
    DATA_PATHS = Path("../data/")

    def __init__(self, datatype, task, algo, cache_dir, results_dir, selection_model_type=None):
        """
        Run a `RandomizedSearchCV(n_iter=25)` coupled with a `RepeatedStratifiedKFold(n_splits=10, n_repeats=10)`
        for every combination of :
            - `algorithm`
            - `task`
            - `datatype`
            - `feature selection strategy`
        """
        self.task = task
        self.datatype = datatype
        self.algo = algo
        self.selection_model_type = selection_model_type
        self.cache_dir = cache_dir
        self.results_dir = results_dir
        
        # edit here
        with open("/Users/thomas/Desktop/datapaths.json", "r") as json_input:
            self.data_paths = json.load(json_input)
        
        self.learn()
        self.save_exp()

    @staticmethod
    def load_pkl(path):
        """
        Load a serialized object from disk.
        Parameters:
        -----------
        path: path to the object.

        Returns:
        --------
        the object.
        """

        return pickle.load(open(path, "rb"))
    
    def save_exp(self):
        """
        Save the grid results, the best index, the best estimator and the selected features on disk.
        Parameters:
        -----------
        obj: the object;
        path: location on disk.
        """
        results_dir = Path(self.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        obj = {
                "results_matrix": self.res,
                "best_index": self.best_index,
                "best_estimator": self.best_estimator,
                "selected_features": self.selected_features,  
              }
        
        res_file = results_dir.joinpath(f"task-{self.task}_datatype-{self.datatype}_algo-{self.algo}_selection-{self.selection_model_type}.pkl")
        with open(res_file, "wb") as f_output:
            pickle.dump(obj, f_output)
    
    @staticmethod
    def keep_common_features(df, n=2):
        """
        Remove features that are not shared by at least n sample(s).
        Parameters:
        -----------
        df: dataframe to filter.
        n: number of samples (default=2).
        
        Returns:
        --------
        the filtered dataframe.
        """
        data_bin = pd.DataFrame(np.where(df > 0, 1, 0),
                                index=list(df.index.values),
                                columns=list(df.columns.values))
        mat_bled = data_bin.loc[:, data_bin.sum() >= n]
        return df.loc[:, mat_bled.columns.values]
                                 
    def dispatch(self):
        """
        Return the complete dataframe used for the learning
        and the corresponding labels.
        
        Returns:
        --------
        out: dataframe.
        y: labels array
        """
        
        df = self.load_pkl(self.data_paths[self.datatype])
        metadata = pd.read_csv(self.data_paths["metadata"], sep=",", index_col=0, header=0)
        out = df.loc[set(metadata[self.task].dropna().index).intersection(df.index)]
        out = self.keep_common_features(out).reindex(sorted(out.index))
        y = np.array(metadata.loc[out.index, self.task])
        return out, y
    
    def learn(self):
        """
        Core implementation.
        """
        
        estimator, estimator_grid = Estimators.hyperparameters_search(self.algo)

        main_pipeline = Pipeline(steps=[
            ('zero_variance_remover', VarianceThreshold(threshold=0)),
            ('scaler', StandardScaler()), # placeholder
            ('selector', SelectFromModel(estimator=BaseEstimator())), # placeholder
            ('estimator', estimator)],
            memory=self.cache_dir)
        
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "roc_auc": make_scorer(roc_auc_score),
            "f1": make_scorer(f1_score),
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score)
        }
        
        param_grid = FeatureSelectionUtils.selector_param_grid(self.selection_model_type)
        param_grid.update(estimator_grid)
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=Learning.RANDOM_STATE)
        grid = RandomizedSearchCV(estimator=main_pipeline,
                                  param_distributions=param_grid,
                                  n_iter=25,
                                  scoring=scoring,
                                  n_jobs=-1,
                                  refit="balanced_accuracy",
                                  cv=cv,
                                  verbose=10,
                                  random_state=Learning.RANDOM_STATE,
                                  return_train_score=True,
                                  error_score=0)
        
        X, y = self.dispatch()
        grid.fit(X, y)
        
        self.res = pd.DataFrame.from_dict(grid.cv_results_)
        self.best_index = grid.best_index_
        self.best_estimator = grid.best_estimator_
        self.selected_features = X.columns[grid.best_estimator_.named_steps['selector'].get_support()]
