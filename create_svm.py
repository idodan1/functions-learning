#!/usr/bin/env python3

import numpy as np
from sklearn import svm

from functions import in_same_family


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
    del cfg["sample"]
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


def predict(cfg, train_x, train_y, test_data_x, test_data_y, dim):
    """
    creates and validates/test an svm model
    returns the percentage of successful predictions on validation/test data.
    """
    train_x = np.array(train_x)[cfg["sample"]]  # leaves only the points chosen for this model
    train_y = np.array(train_y)[cfg["sample"]]
    train_x = train_x[:int(len(train_x) * float(cfg["percent_of_points"])), :dim * cfg["num_of_points"]]
    train_y = train_y[:int(len(train_y) * float(cfg["percent_of_points"]))]
    test_data_x = np.array(test_data_x)[:, :dim*cfg["num_of_points"]]

    clf = svm_from_cfg(cfg)
    clf.fit(train_x, train_y)
    prediction = clf.predict(test_data_x)

    counter = 0
    family_counter = 0
    for i in range(len(prediction)):
        if prediction[i] == test_data_y[i]:
            counter += 1
        elif in_same_family(prediction[i], test_data_y[i]):
            family_counter += 1
    return counter/len(test_data_y)*100, (counter + family_counter)/len(test_data_y)*100







