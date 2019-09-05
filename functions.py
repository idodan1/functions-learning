#!/usr/bin/env python3

import numpy as np
import csv
import pandas as pd


def find_index(values_cum_sum, alpha):
    index = 0
    while alpha > values_cum_sum[index]:
        index += 1
    return index


def get_data_from_file(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = np.array(list(reader))

    df = pd.read_csv(file_name)
    columns = list(range(df.shape[1] - 1))
    columns.append("y")
    df.columns = columns
    data_x = data[:, :len(data[0])-1]
    data_y = data[:, len(data[0])-1]
    return data_x, data_y, df


def create_data_for_model(data_x_old, data_y_old, train_ratio, valid_ratio, test_ratio, members, df):
    data_x, data_y, df = leave_only(data_x_old, data_y_old, members, df)
    length = len(data_y)

    train_x = data_x[:int(length*train_ratio)]
    train_y = data_y[:int(length*train_ratio)]
    validation_x = data_x[int(length*train_ratio):int(length*valid_ratio)]
    validation_y = data_y[int(length*train_ratio):int(length*valid_ratio)]
    test_x = data_x[int(length*valid_ratio):int(length*test_ratio)]
    test_y = data_y[int(length*valid_ratio):int(length*test_ratio)]

    train_x = np.array(train_x).astype(np.float)
    train_y = np.array(train_y).astype(np.float).astype(np.int64)
    validation_x = np.array(validation_x).astype(np.float)
    validation_y = np.array(validation_y).astype(np.float).astype(np.int64)
    test_x = np.array(test_x).astype(np.float)
    test_y = np.array(test_y).astype(np.float).astype(np.int64)
    return train_x, train_y, validation_x, validation_y, test_x, test_y


def leave_only(data_x, data_y, members, df):
    df = df.loc[df['y'].isin(members)]
    data_y = np.array(data_y).astype(np.float)
    index = [True if data_y[i] in members else False for i in range(len(data_y))]
    new_x = []
    new_y = []
    for i in range(len(index)):
        if index[i]:
            new_x.append(data_x[i])
            new_y.append(data_y[i])
    return np.array(new_x), np.array(new_y), df


def in_same_family(x, y):
    if x <= 5 and y <= 5:
        return True
    elif 9 >= x >= 6 and 9 >= y >= 6:
        return True
    elif 14 >= x >= 10 and 14 >= y >= 10:
        return True
    elif 19 >= x >= 15 and 19 >= y >= 15:
        return True
    elif 24 >= x >= 20 and 24 >= y >= 20:
        return True
    return False





