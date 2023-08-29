import sys
import numpy as np
import pickle
import pandas as pd
import re

sys.path.append("../mlcog")


saved_exp = pickle.load(open(sys.argv[1], "rb"))
best_estimator = saved_exp["best_estimator"].named_steps['estimator']
model_type = type(best_estimator).__name__
model_params = best_estimator.get_params()

selected_features = saved_exp["selected_features"]

def throw_features_stdout(importances):
    importances_abs = abs(importances)
    sorted_indices = np.argsort(importances_abs)[::-1]
    sorted_feature_names = [selected_features[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]
    for name, importance in zip(sorted_feature_names, sorted_importances):
        print(f"{name}\t{importance}")

if model_type == "SVC":
    raise NotImplementedError("SVM_rbf has no straightforward interpretation method.")

elif model_type == "LinearSVC":
    importances = best_estimator.coef_[0]
    throw_features_stdout(importances)

elif model_type == "LogisticRegression":
    importances = best_estimator.coef_[0]
    throw_features_stdout(importances)

elif model_type == "DecisionTreeClassifier":
    importances = best_estimator.feature_importances_
    throw_features_stdout(importances)

elif model_type == "RandomForestClassifier":
    importances = best_estimator.feature_importances_
    throw_features_stdout(importances)

elif model_type == "XGBClassifier":
    importances = best_estimator.feature_importances_
    throw_features_stdout(importances)

elif model_type == "SetCoveringMachineClassifier":
    model = str(best_estimator.model_)
    pattern = r"X\[\d+\]"

    def replace_match(match):
        number = int(match.group(0)[2:-1])
        new_value = selected_features[number]
        return new_value
    
    model_with_names = re.sub(pattern, replace_match, model)
    print(model_with_names)

elif model_type == "MultinomialNB":
    pos_class_prob = best_estimator.feature_log_prob_[1,:]
    pos_class_prob_ind_sorted = pos_class_prob.argsort()[::-1]
    for ind in pos_class_prob_ind_sorted:
        print(f"{selected_features[ind]}\t{pos_class_prob[ind]}")
