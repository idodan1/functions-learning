#!/usr/bin/env python3

import numpy as np
from sklearn import svm

from functions import in_same_family


def create_configuration_space_svm(num_of_data_points):
    configuration_space = dict()
    configuration_space["kernel"] = ["rbf", "sigmoid"]
    configuration_space["C"] = [0.001, 1000.0]
    configuration_space["shrinking"] = ["true", "false"]
    configuration_space["degree"] = [1, 5]
    configuration_space["coef0"] = [0.0, 10.0]
    configuration_space["gamma"] = ["auto", "value"]
    configuration_space["gamma_value"] = [0.0001, 8.0]  # only if gamma is value
    configuration_space["percent_of_points"] = [0.5, 1]
    configuration_space["num_of_points"] = [num_of_data_points-5, num_of_data_points]
    return configuration_space


def svm_from_cfg(cfg_with_extra_features):
    """ Creates a SVM based on a configuration.

    Parameters:
    -----------
    cfg_with_extra_features - a configuration for a new svm model.

    Returns:
    --------
    An svm model ready for validation.
    """
    cfg = dict(cfg_with_extra_features)
    del cfg["percent_of_points"]
    del cfg["num_of_points"]
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma_value" not in cfg.keys():
        cfg["gamma_value"] = 0.000001
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=42)

    return clf


def predict_svm(df_pop, df_train, df_valid, df_test, dim, members, in_training=False):
    """
        creates and validates/test an svm model
        returns the percentage of successful predictions on validation/test data.
        """
    if in_training:
        df_test_for_func = df_valid.copy()
    else:
        df_test_for_func = df_test
    train_x = np.asmatrix(df_train.drop(columns=['y']).values)
    train_y = np.array(df_train['y'].values)
    test_y = np.array(df_test_for_func['y'].values)
    results = []
    results_family = []
    length = len(test_y)
    for j in range(len(df_pop)):
        #  create dict from row for creating a svm model
        cfg = dict(zip(df_pop.columns.tolist(), df_pop[j:j+1].values.tolist()[0]))
        train_x_cfg = train_x[:int(len(train_x) * float(cfg["percent_of_points"])), :dim * cfg["num_of_points"]]
        train_y_cfg = train_y[:int(len(train_y) * float(cfg["percent_of_points"]))]
        test_x = np.asmatrix(df_test_for_func.drop(columns=['y']).values)[:, :dim*cfg["num_of_points"]]

        clf = svm_from_cfg(cfg)
        clf.fit(train_x_cfg, train_y_cfg)
        prediction = clf.predict(test_x)

        counter = 0
        family_counter = 0
        for i in range(len(prediction)):
            if prediction[i] == test_y[i]:
                counter += 1
            elif in_same_family(prediction[i], test_y[i], members):
                family_counter += 1
        results.append(counter*100/length)
        results_family.append((counter+family_counter)*100/length)
    df_pop = df_pop.assign(results=results, results_family=results_family)
    return df_pop


