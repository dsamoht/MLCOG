"""Estimators instances and hyperparameters grids."""
from sklearn.utils.fixes import loguniform
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from pyscm import SetCoveringMachineClassifier
#from randomscm.randomscm import RandomScmClassifier
from xgboost import XGBClassifier


class Estimators:
    """
    Estimators instances and hyperparameters grids.
    """

    RANDOM_STATE = 0

    @staticmethod
    def hyperparameters_search(algo):
        """
        Instantiate an estimator corresponding to a given `algo`
     
        Parameters:
        -----------
        algo: algorithm.

        Returns:
        --------
        tuple containing : 
            [0] estimator instance
            [1] hyperparameters grid 
        """

        estimators_instances = {'SVM_rbf': svm.SVC(),
                                'SVM_L1': svm.LinearSVC(),
                                'SVM_L2': svm.LinearSVC(),
                                'Logistic_Regression_L1': LogisticRegression(),
                                'Logistic_Regression_L2': LogisticRegression(),
                                'Decision_tree': DecisionTreeClassifier(),
                                'random_forest': RandomForestClassifier(),
                                'SCM': SetCoveringMachineClassifier(),
                                #'Random_SCM': RandomScmClassifier
                                'XGBoost': XGBClassifier(),
                                'MNB': MultinomialNB()
                               }

        c_params = loguniform(0.001, 100)

        if algo == 'SVM_rbf':

            estimator_grid = {"estimator__C": c_params,
                              "estimator__max_iter": [-1],
                              "estimator__class_weight": ["balanced"],
                              "estimator__kernel": ["rbf"],
                              "estimator__gamma": ["scale"], 
                              "estimator__probability": [True],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }

        elif algo == 'SVM_L1':

            estimator_grid = {"estimator__C": c_params,
                              "estimator__max_iter": [1000],
                              "estimator__class_weight": ["balanced"],
                              "estimator__penalty": ["l1"],
                              "estimator__loss": ["squared_hinge"], 
                              "estimator__dual": [False],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }

        elif algo == "SVM_L2":

            estimator_grid = {"estimator__C": c_params,
                              "estimator__max_iter": [1000],
                              "estimator__class_weight": ["balanced"],
                              "estimator__penalty": ["l2"],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }

        elif algo == "Logistic_Regression_L1":

            estimator_grid = {"estimator__C": c_params,
                              "estimator__solver": ["liblinear"],
                              "estimator__class_weight": ["balanced"],
                              "estimator__penalty": ["l1"],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }
    
        elif algo == "Logistic_Regression_L2":
            
            estimator_grid = {"estimator__C": c_params,
                              "estimator__class_weight": ["balanced"],
                              "estimator__penalty": ["l2"],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }

        elif algo == "Decision_tree":
            
            estimator_grid = {"estimator__max_depth": [1, 3, 5, 10, 25],
                              "estimator__class_weight": ["balanced"],
                              "estimator__min_samples_split": [2, 5, 10],
                              "estimator__min_samples_leaf": [1, 2, 4],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }

        elif algo == "random_forest":
            
            estimator_grid = {"estimator__max_depth": [1, 3, 5, 10, 25],
                              "estimator__class_weight": ["balanced"],
                              "estimator__min_samples_split": [2, 5, 10],
                              "estimator__min_samples_leaf": [1, 2, 4],
                              "estimator__n_estimators": [500],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }
            
        elif algo == "SCM":
            
            estimator_grid = {"estimator__p": [0.5, 1, 2],
                              "estimator__max_rules": [1, 2, 3, 4, 5],
                              "estimator__model_type": ["conjunction", "disjunction"],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }
    
        elif algo == "Random_SCM":
            
            estimator_grid = {"estimator__n_estimators": [30, 100],
                              "estimator__max_samples": [0.6, 0.85],
                              "estimator__max_features": [0.6, 0.85],
                              "estimator__p_options": [ [0.1], [0.316], [0.45], [0.562], [0.65], [0.85], [1.0], [2.5], [4.39], [5.623], [7.623], [10.0] ],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }
        
        elif algo == "XGBoost":
            
            estimator_grid = {"estimator__eta": [0.001, 0.01],
                              "estimator__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
                              "estimator__max_depth": [1, 3, 5, 10, 25],
                              "estimator__subsample": [0.5, 0.75, 1],
                              "estimator__random_state": [Estimators.RANDOM_STATE]
                             }
            
        elif algo == "MNB":
            
            estimator_grid = {"estimator__alpha": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                              "estimator__fit_prior": [True, False]
                             }

        return estimators_instances[algo], estimator_grid
