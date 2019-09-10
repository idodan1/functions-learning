# !/usr/bin/env python

import numpy as np
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential
import logging
import warnings
import sys
from functions import in_same_family
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.WARNING)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("../..")


def create_configuration_space_net(num_of_data_points, max_layers=10, max_neuron_in_layer=100):
    configuration_space = dict()
    configuration_space["num_of_layers"] = [2, max_layers]
    configuration_space["percent_of_points"] = [0.1, 1.0]
    configuration_space["learning_rate"] = [0.001, 1]
    configuration_space["num_of_points"] = [num_of_data_points-10, num_of_data_points]
    configuration_space["optimizer"] = ["adam", "sgd", "adadelta"]
    configuration_space["epochs"] = [2, 5]
    for i in range(max_layers):
        configuration_space['layer'+str(i)+' num of neuron'] = [10, max_neuron_in_layer]
        #  this will be multiplied by 10
        configuration_space['layer'+str(i)+' activation'] = ["relu", "sigmoid", "elu", "selu"]
    return configuration_space


# this function is needed for later
def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s


def build_model(configuration, num_of_functions, feature_len):  # feature_len = len(x_train[0])
    # we need to clear the graph
    tf.logging.set_verbosity(tf.logging.ERROR)

    s = reset_tf_session()

    model = Sequential()  # it is a feed-forward network without loops like in RNN
    model.add(Dense(configuration['layer0 num of neuron'] * 10, input_shape=(feature_len,)))
    # the first layer must specify the input shape (replacing placeholders)
    try:
        model.add(Activation(configuration['layer0 activation']))
    except:
        model.add(Activation("sigmoid"))
    for i in range(1, configuration["num_of_layers"]):
        try:
            model.add(Dense(configuration['layer'+str(i)+' num of neuron']*10,
                            activation=configuration['layer'+str(i)+' activation']))
        except:
            model.add(Dense(configuration[str('layer'+str(i)+' num of neuron')]*10,
                            activation="sigmoid"))

    model.add(Dense(num_of_functions))  # num of different functions
    model.add(Activation('softmax'))

    # compile model
    model.compile(
        loss='categorical_crossentropy',  # this is our cross-entropy
        optimizer=configuration["optimizer"],
        metrics=['accuracy']  # report accuracy during training
    )

    return model


def fit_model(model, x_train, y_train_oh, x_valid, y_valid_oh, epochs):
    history = model.fit(
        x_train,
        y_train_oh,
        batch_size=512,
        epochs=epochs,
        validation_data=(x_valid, y_valid_oh),
        verbose=0
    )

    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    return accuracy, val_accuracy


def predict_net(df_pop, df_train, df_valid, df_test, dim, members, in_training=False):
    """
    in this function we create a model from configuration, train it and test it on the test data.
    It should be considered to use the accuracy value that fit_model returns which is not used at the moment
    and maybe avoiding some calculations.
    """
    train_x = np.asmatrix(df_train.drop(columns=['y']).values)
    valid_x = np.asmatrix(df_valid.drop(columns=['y']).values)
    test_x = np.asmatrix(df_test.drop(columns=['y']).values)

    train_y = np.array(df_train['y'].values)
    valid_y = np.array(df_valid['y'].values)
    test_y = np.array(df_test['y'].values)

    num_of_functions = len(df_test.y.unique())
    length = len(test_y)

    results = []
    results_family = []

    valid_y_oh = keras.utils.to_categorical(valid_y, num_of_functions+1)
    valid_y_oh = valid_y_oh[:, 1:]

    for j in range(len(df_pop)):
        cfg = dict(zip(df_pop.columns.tolist(), df_pop[j:j + 1].values.tolist()[0]))
        train_x_cfg = train_x[:int(len(train_x) * float(cfg["percent_of_points"])), :dim * cfg["num_of_points"]]
        valid_x_cfg = valid_x[:, :dim * cfg["num_of_points"]]
        test_x_cfg = test_x[:, :dim * cfg["num_of_points"]]

        train_y_cfg = train_y[:int(len(train_y) * float(cfg["percent_of_points"]))]
        train_y_cfg_oh = keras.utils.to_categorical(train_y_cfg, num_of_functions+1)
        train_y_cfg_oh = train_y_cfg_oh[:, 1:]

        feature_len = train_x_cfg[0].shape[1]
        model = build_model(cfg, num_of_functions, feature_len)
        acc, val_acc = fit_model(model, train_x_cfg, train_y_cfg_oh, valid_x_cfg, valid_y_oh, cfg["epochs"])
        if in_training:
            results.append(val_acc[-1] * 100)
        else:
            prediction = np.argmax(model.predict(test_x_cfg), axis=1)
            counter_correct = 0
            counter_same_family = 0
            for i in range(len(test_y)):
                if test_y[i] == (prediction[i] + 1):  # need to add one because prediction starts from 0
                    counter_correct += 1
                elif in_same_family(int(test_y[i]), int(prediction[i]+1), members):
                    counter_same_family += 1
            results.append(counter_correct * 100 / length)
            results_family.append((counter_correct + counter_same_family) * 100 / length)
    if in_training:
        results_family = [0]*len(results)
    df_pop = df_pop.assign(results=results, results_family=results_family)
    return df_pop
