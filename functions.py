#!/usr/bin/env python3

from random import shuffle
import pandas as pd


def find_index(values_cum_sum, alpha):
    index = 0
    while alpha > values_cum_sum[index]:
        index += 1
    return index


def get_data_from_file(file_name, num_of_files_for_data):
    shuffle(file_name)  # every time we will get different points
    df = pd.read_csv(file_name[0])
    columns = list(range(df.shape[1] - 1))
    columns.append("y")
    df.columns = columns
    # for i in range(1, len(file_name)):
    for i in range(1, num_of_files_for_data):
        try:
            temp = pd.read_csv(file_name[i])
            temp.columns = columns
            df = df.append(temp, ignore_index=True)
        except:
            break
    return df


def create_data_for_model(train_ratio, validation_ratio, test_ratio, members, df):
    df = leave_only(members, df)
    df = df.apply(pd.to_numeric)
    #  we need to change the data_y to contain sequential members for one hot vectors
    change_members = range(1, len(members)+1)
    df['y'] = [change_members[members.index(number)] for number in df['y'].values]
    length = df.shape[0]  # num of rows in df

    df_train = df[0:int(length*train_ratio)]
    df_valid = df[int(length*train_ratio):int(length*validation_ratio)]
    df_test = df[int(length*validation_ratio):int(length*test_ratio)]

    return df_train, df_valid, df_test


def leave_only(members, df):
    df = df.loc[df['y'].isin(members)]
    return df


def in_same_family(x_before, y_before, members):
    x = members[x_before-1]
    y = members[y_before-1]
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


def add_prediction(wrong_predictions_df, function_num, wrong_prediction, wrong=False):
    if wrong:
        wrong_predictions_df.at[str(function_num), 'wrong_prediction'] = wrong_predictions_df.loc[
                                                                             str(function_num), [
                                                                                 'wrong_prediction']].values[0] + 1.0
        wrong_predictions_df.at[str(function_num), wrong_prediction] = wrong_predictions_df.loc[
                                                                       str(function_num), [
                                                                           wrong_prediction]].values[0] + 1.0
    else:
        wrong_predictions_df.at[str(function_num), 'right_prediction'] = wrong_predictions_df.loc[
                                                                             str(function_num), [
                                                                                 'right_prediction']].values[0] + 1.0



