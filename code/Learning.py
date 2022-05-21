import numpy as np
import os
import pickle
from pyscm import SetCoveringMachineClassifier
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from randomscm.randomscm import RandomScmClassifier


class Learning:
    def __init__(self, dataset, datatype, seed, directory, algo=False, saveModel=False):
        self.dataset = dataset
        self.datatype = datatype
        self.seed = seed
        self.directory = directory
        self.saveModel = saveModel
        self.algo = algo
        self.algo_caller = {
            'SVM_rbf': svm.SVC,
            'SVM_L1': svm.LinearSVC,
            'SVM_L2': svm.LinearSVC,
            'Logistic_Regression_L1': LogisticRegression,
            'Logistic_Regression_L2': LogisticRegression,
            'Decision_tree': DecisionTreeClassifier,
            'random_forest': RandomForestClassifier,
            'SCM': SetCoveringMachineClassifier,
            'Random_SCM': RandomScmClassifier
        }

        self.learn()

    def modelPickler(self, trainedModel, algo, i, Params_iteration):
        """
        Save a serialized model on disk in ./SAVED_MODEL/
        Parameters:
        -----------
        trainedModel: a fitted model.
        algo: algorithm.
        i: 10-fold CV iteration number (0-9).
        Params_iteration: hyperparameters search iteration number.
        """

        with open(f"SAVED_MODELS/{self.dataset}-{self.datatype}-{algo}-{self.seed}_iter-{i}-{Params_iteration}_model.pkl", "wb") as f:

            pickle.dump(trainedModel, f)

    def xtrainPickler(self, arr, i):
        """
        Save a serialized array on disk in ./SAVED_MODEL/
        Parameters:
        -----------
        arr: array.
        i: 10-fold CV iteration number (0-9).
        Params_iteration: hyperparameters search iteration number.
        """

        with open(f"SAVED_MODELS/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_XTRAIN.pkl", "wb") as f:

            pickle.dump(arr, f)

    def hyperparametersSearch(self, algo):
        """
        Instantiate a model for every combination of hyperparameters
        Parameters:
        -----------
        algo: algorithm.

        Returns:
        --------
        models: list of instantiated models
        """

        models = []
        C_params = [0.0001, 0.001, 0.01, 0.1, 0.25,
                    0.5, 0.8, 0.9, 1, 10, 100, 500, 1000]

        if algo == 'SVM_rbf':
            for C in C_params:
                models.append(
                    self.algo_caller[algo](
                        C=C,
                        max_iter=-1,
                        class_weight='balanced',
                        gamma='scale',
                        probability=True,
                        random_state=1,
                        kernel='rbf'))
        elif algo == 'SVM_L1':
            for C in C_params:
                for max_iter in [1000]:
                    models.append(
                        self.algo_caller[algo](
                            C=C,
                            max_iter=max_iter,
                            class_weight='balanced',
                            penalty='l1',
                            random_state=1,
                            dual=False,
                            loss='squared_hinge'))
        elif algo == "SVM_L2":
            for C in C_params:
                for max_iter in [1000]:
                    models.append(
                        self.algo_caller[algo](
                            C=C,
                            max_iter=max_iter,
                            class_weight='balanced',
                            penalty='l2',
                            random_state=1))
        elif algo == "Logistic_Regression_L1":
            for C in C_params:
                models.append(
                    self.algo_caller[algo](
                        C=C,
                        class_weight='balanced',
                        penalty='l1',
                        random_state=1,
                        solver='liblinear'))
        elif algo == "Logistic_Regression_L2":
            for C in C_params:
                models.append(
                    self.algo_caller[algo](
                        C=C,
                        class_weight='balanced',
                        penalty='l2',
                        random_state=1))
        elif algo == "Decision_tree":
            for depth in [1, 3, 5, 10, 25]:
                for min_samples_split in [2, 5, 10]:
                    for min_samples_leaf in [1, 2, 4]:
                        models.append(
                            self.algo_caller[algo](
                                max_depth=depth,
                                class_weight='balanced',
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=1))
        elif algo == "random_forest":
            for depth in [1, 5, 10, 25]:
                for min_samples_split in [2, 5, 10]:
                    for min_samples_leaf in [1, 2, 4]:
                        models.append(
                            self.algo_caller[algo](
                                max_depth=depth,
                                class_weight='balanced',
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=1,
                                n_estimators=500))
        elif algo == "SCM":
            for p in [0.5, 1, 2]:
                for max_rules in [1, 2, 3, 4, 5]:
                    for model_type in ["conjunction", "disjunction"]:
                        models.append(
                            self.algo_caller[algo](
                                p=p,
                                max_rules=max_rules,
                                model_type=model_type,
                                random_state=1))
        elif algo == "Random_SCM":
            for n_estimators in [30, 100]:
                for max_samples in [0.6, 0.85]:
                    for max_features in [0.6, 0.85]:
                        for p_options in [ [0.1], [0.316], [0.45], [0.562], [0.65], [0.85],
                                           [1.0], [2.5], [4.39], [5.623], [7.623], [10.0] ]:
                            models.append(
                                self.algo_caller[algo](
                                    n_estimators=n_estimators,
                                    max_samples=max_samples,
                                    max_features=max_features,
                                    p_options=p_options,
                                    random_state=1))

        return models

    def computeScores(self, y_test, y_test_pred, y_train, y_train_pred, fpr_TRAIN, tpr_TRAIN, fpr_TEST, tpr_TEST):
        """
        Compute metrics for a given set of prediction results
        Parameters:
        -----------
        y_test: test set labels.
        y_test_pred: test predictions.
        y_train: train set labels.
        y_train_pred: train set predictions.
        fpr_TRAIN: train set false positive rate.
        tpr_TRAIN: train set true positive rate.
        fpr_TEST: test set false positive rate.
        tpr_TEST: test set true positive rate.

        Returns:
        --------
        results: dictionnary
        """

        results = {}
        results["bacc_TEST"] = balanced_accuracy_score(y_test, y_test_pred)
        results["acc_TEST"] = accuracy_score(y_test, y_test_pred)
        results["f1_TEST"] = f1_score(y_test, y_test_pred)
        results["precision_TEST"] = precision_score(y_test, y_test_pred)
        results["recall_TEST"] = recall_score(y_test, y_test_pred)
        results["auc_TEST"] = auc(fpr_TEST, tpr_TEST)

        results["bacc_TRAIN"] = balanced_accuracy_score(y_train, y_train_pred)
        results["acc_TRAIN"] = accuracy_score(y_train, y_train_pred)
        results["f1_TRAIN"] = f1_score(y_train, y_train_pred)
        results["precision_TRAIN"] = precision_score(y_train, y_train_pred)
        results["recall_TRAIN"] = recall_score(y_train, y_train_pred)
        results["auc_TRAIN"] = auc(fpr_TRAIN, tpr_TRAIN)

        return results

    def learn(self):
        """
        10-fold cross-validation with hyperparameters search
        for a given combination of dataset, datatype and algorithm.
        Save the results dictionnary on disk in ./RESULT_DICTS.
        """

        if not self.algo:
            algos = [
                "SVM_L1",
                "SVM_L2",
                "SVM_rbf",
                "Logistic_Regression_L1",
                "Logistic_Regression_L2",
                "Decision_tree",
                "random_forest",
                "SCM",
                "Random_SCM"
                ]
        else:
            algos = [self.algo]

        for algo in algos:
            main_res = {}
            for i in range(10):

                X_train = pickle.load(
                    open(
                        f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_XTRAIN.pkl",
                        "rb"))
                X_test = pickle.load(
                    open(
                        f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_XTEST.pkl",
                        "rb"))
                y_train = pickle.load(
                    open(
                        f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_YTRAIN.pkl",
                        "rb"))
                y_test = pickle.load(
                    open(
                        f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_YTEST.pkl",
                        "rb"))

                bAccs_TRAIN, bAccs_TEST = [], []
                Accs_TRAIN, Accs_TEST = [], []
                AUCs_TRAIN, AUCs_TEST = [], []
                F1s_TRAIN, F1s_TEST = [], []
                Precisions_TRAIN, Precisions_TEST = [], []
                Recalls_TRAIN, Recalls_TEST = [], []

                params = []
                for j, m in enumerate(self.hyperparametersSearch(algo)):

                    m.fit(X_train, y_train)
                    if self.saveModel:
                        self.modelPickler(m, algo, i, j)
                        self.xtrainPickler(X_train, i)

                    y_train_pred = m.predict(X_train)
                    y_test_pred = m.predict(X_test)

                    if algo in ["SVM_L1", "SVM_L2"]:
                        y_train_score = m.decision_function(X_train)
                        y_test_score = m.decision_function(X_test)
                        fpr_TRAIN, tpr_TRAIN, _ = roc_curve(
                            y_train, y_train_score, pos_label=1)
                        fpr_TEST, tpr_TEST, _ = roc_curve(
                            y_test, y_test_score, pos_label=1)

                    else:
                        y_train_proba = m.predict_proba(X_train)
                        y_test_proba = m.predict_proba(X_test)
                        fpr_TRAIN, tpr_TRAIN, _ = roc_curve(
                            y_train, y_train_proba[:, 1], pos_label=1)
                        fpr_TEST, tpr_TEST, _ = roc_curve(
                            y_test, y_test_proba[:, 1], pos_label=1)

                    res = self.computeScores(y_test, y_test_pred, y_train, y_train_pred,
                                              fpr_TRAIN, tpr_TRAIN, fpr_TEST, tpr_TEST)
                    bAccs_TRAIN.append(res["bacc_TRAIN"])
                    bAccs_TEST.append(res["bacc_TEST"])
                    Accs_TRAIN.append(res["acc_TRAIN"])
                    Accs_TEST.append(res["acc_TEST"])
                    AUCs_TRAIN.append(res["auc_TRAIN"])
                    AUCs_TEST.append(res["auc_TEST"])
                    F1s_TRAIN.append(res["f1_TRAIN"])
                    F1s_TEST.append(res["f1_TEST"])
                    Precisions_TRAIN.append(res["precision_TRAIN"])
                    Precisions_TEST.append(res["precision_TEST"])
                    Recalls_TRAIN.append(res["recall_TRAIN"])
                    Recalls_TEST.append(res["recall_TEST"])

                    params.append(m.get_params())

                main_res["params"] = params
                main_res[f"split{i}_train_balanced_accuracy"] = bAccs_TRAIN
                main_res[f"split{i}_train_accuracy"] = Accs_TRAIN
                main_res[f"split{i}_train_roc_auc"] = AUCs_TRAIN
                main_res[f"split{i}_train_f1"] = F1s_TRAIN
                main_res[f"split{i}_train_precision"] = Precisions_TRAIN
                main_res[f"split{i}_train_recall"] = Recalls_TRAIN
                main_res[f"split{i}_test_balanced_accuracy"] = bAccs_TEST
                main_res[f"split{i}_test_accuracy"] = Accs_TEST
                main_res[f"split{i}_test_roc_auc"] = AUCs_TEST
                main_res[f"split{i}_test_f1"] = F1s_TEST
                main_res[f"split{i}_test_precision"] = Precisions_TEST
                main_res[f"split{i}_test_recall"] = Recalls_TEST

            with open(f"RESULT_DICTS/dataset-{self.dataset}_datatype-{self.datatype}_algo-{algo}_seed-{self.seed}_results_dict.pkl", "wb") as f:
                pickle.dump(main_res, f)
