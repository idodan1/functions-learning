import numpy as np
import os
import tensorflow as tf
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
from functions import in_same_family, add_prediction
from create_net import reset_tf_session, fit_model, create_configuration_space_net


def create_configuration_space_cnn(num_of_data_points, max_layers=10, max_neuron_in_layer=100):
    configuration_space = create_configuration_space_net(num_of_data_points, max_layers, max_neuron_in_layer)
    configuration_space['kernel_size'] = [2, 11]
    configuration_space['num_of_conv'] = [1, max_layers]
    return configuration_space


def build_model_conv(configuration, num_of_functions, feature_len):  # feature_len = len(x_train[0])
    # we need to clear the graph
    tf.logging.set_verbosity(tf.logging.ERROR)

    s = reset_tf_session()

    model = Sequential()  # it is a feed-forward network without loops like in RNN
    model.add(Conv1D(int(configuration['layer0 num of neuron'] * 10), kernel_size=int(configuration['kernel_size']),
                     input_shape=(int(feature_len), 1)))
    try:
        model.add(Activation(configuration['layer0 activation']))
    except:
        model.add(Activation("sigmoid"))
    model.add(Flatten())
    for i in range(1, int(configuration["num_of_layers"])):
        try:
            model.add(Dense(int(configuration['layer'+str(i)+' num of neuron']*10),
                            activation=configuration['layer'+str(i)+' activation']))
            # model.add(Conv1D(configuration['layer' + str(i) + ' num of neuron'] * 10, kernel_size=configuration['kernel_size'],
            #                 activation=configuration['layer' + str(i) + ' activation']))
            # model.add(Flatten())
        except:
            model.add(Dense(configuration[str('layer'+str(i)+' num of neuron')]*10,
                            activation="sigmoid"))

    model.add(Dense(num_of_functions))  # num of different functions
    model.add(Activation('softmax'))

    # compile model
    model.compile(
        loss='categorical_crossentropy',  # this is our cross-entropy
        optimizer='adam',
        metrics=['accuracy']  # report accuracy during training
    )

    return model


def predict_net_conv(df_pop, df_train, df_valid, df_test, dim, members, in_training=False, wrong_predictions_df=None):
    """
    in this function we create a model from configuration, train it and test it on the test data.
    It should be considered to use the accuracy value that fit_model returns which is not used at the moment
    and maybe avoiding some calculations.
    """
    train_x = np.asmatrix(df_train.drop(['y'], axis=1).values)
    valid_x = np.asmatrix(df_valid.drop(['y'], axis=1).values)
    test_x = np.asmatrix(df_test.drop(['y'], axis=1).values)

    train_x = np.expand_dims(train_x, axis=2)
    valid_x = np.expand_dims(valid_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)

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
        train_x_cfg = train_x[:int(len(train_x) * float(cfg["percent_of_points"])), :int(dim * cfg["num_of_points"])]
        valid_x_cfg = valid_x[:, :int(dim * cfg["num_of_points"])]
        test_x_cfg = test_x[:, :int(dim * cfg["num_of_points"])]

        train_y_cfg = train_y[:int(len(train_y) * float(cfg["percent_of_points"]))]
        train_y_cfg_oh = keras.utils.to_categorical(train_y_cfg, num_of_functions+1)
        train_y_cfg_oh = train_y_cfg_oh[:, 1:]

        feature_len = int(train_x_cfg[0].shape[0])
        model = build_model_conv(cfg, num_of_functions, feature_len)
        acc, val_acc = fit_model(model, train_x_cfg, train_y_cfg_oh, valid_x_cfg, valid_y_oh, int(cfg["epochs"]))
        if in_training:
            results.append(val_acc[-1] * 100)
        else:
            prediction = np.argmax(model.predict(test_x_cfg), axis=1)
            counter_correct = 0
            counter_same_family = 0
            for i in range(len(test_y)):
                if test_y[i] == (prediction[i] + 1):  # need to add one because prediction starts from 0
                    counter_correct += 1
                    add_prediction(wrong_predictions_df, int(test_y[i]), int(prediction[i]+1))
                else:
                    add_prediction(wrong_predictions_df, int(test_y[i]), int(prediction[i]+1), wrong=True)
                    if in_same_family(int(test_y[i]), int(prediction[i]+1), members):
                        counter_same_family += 1

            results.append(counter_correct * 100 / length)
            results_family.append((counter_correct + counter_same_family) * 100 / length)
    if in_training:
        results_family = [0]*len(results)
    df_pop = df_pop.assign(results=results, results_family=results_family)
    return df_pop

