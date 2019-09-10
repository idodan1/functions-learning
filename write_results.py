#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd


def write_results(df_pop, dir_str, members, list_of_best_in_each_iter, algorithm_name):

    # save results in a folder. father_folder_name == group_size. if not exist, create it.
    dir_str = dir_str + "/group size = " + str(len(members))
    if not os.path.exists(dir_str):
        os.mkdir(dir_str)

    # create a folder for the specific group of functions we are testing
    dir_str = dir_str + "/" + str(members)
    if not os.path.exists(dir_str):
        os.mkdir(dir_str)

    df = pd.DataFrame(list_of_best_in_each_iter, columns=["right prediction"])
    df.to_csv(str(dir_str + "/list_of_best_"+algorithm_name+".csv"))

    df_pop.to_excel(str(dir_str + '/'+algorithm_name+'.xlsx'))


def make_parameter_string(pop_size, num_of_iter, train_ratio, validation_ratio, test_ratio):

    parameter_string = "pop_size = " + str(pop_size)

    parameter_string += " num_of_iter= " + str(num_of_iter)

    parameter_string += " train_ratio= " + str(train_ratio)

    parameter_string += " validation_ratio= " + str(validation_ratio)

    parameter_string += " test_ratio= " + str(test_ratio) + "\n"

    return parameter_string


def create_date_str(date):

    new_date = date[:4] + "." + date[4:6] + "." + date[6:8] + "-" + date[9:11] + "." + date[11:13] + "." + date[13:15]

    return new_date



















