#!/usr/bin/env python3

import pip
from ga import iterate
from functions import get_data_from_file, create_data_for_model
from create_net import create_configuration_space_net, predict_net
from create_svm import create_configuration_space_svm, predict_svm
from write_results import create_date_str, write_results, make_parameter_string
import os
import random
import datetime
import pandas as pd

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

if __name__ == "__main__":
    install('openpyxl')
    
    results_dir_str = "./results_dim10/" + create_date_str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(results_dir_str)
    dir_list = os.listdir('./points')
    df = get_data_from_file([str('./points/' + file_name) for file_name in dir_list])
    dim = 10
    num_of_data_points = int(df.shape[1]/(dim+1))
    pop_size = 2
    num_of_iter = 2
    train_ratio = 0.5
    validation_ratio = 0.7
    test_ratio = 1
    mutation_min_alpha = 0.2
    mutation_delta = 0.1
    parameter_string = make_parameter_string(pop_size, num_of_iter, train_ratio, validation_ratio, test_ratio, len(df))
    f = open(results_dir_str + "/parameters", "w")
    f.write(parameter_string)
    f.close()
    results_df = pd.DataFrame(columns=['algorithm', 'group size', 'members', 'best result', 'best result family'])
    
    for k in range(1):
        for i in range(10, 11):
          try:
            print("group size equals " + str(i))
            for j in range(1):
                members = random.sample(range(1, 25), i)
                df_train, df_valid, df_test = create_data_for_model(train_ratio, validation_ratio,
                                                                    test_ratio, members, df)
                len_data = df.shape[0]  # num of rows in df

                """
                looks for the best model using NN
                """
                df_pop, list_of_best_in_each_iter = iterate(pop_size, num_of_data_points, num_of_iter, dim,
                                                            mutation_min_alpha,mutation_delta, df_train, df_valid,
                                                            df_test, create_configuration_space_net, predict_net,
                                                            members)

                results_df = results_df.append(pd.DataFrame([['NN', i, str(members), df_pop['results'].max(),
                                                              df_pop['results_family'].max()]],
                                                            columns=results_df.columns), ignore_index=True)
                write_results(df_pop, results_dir_str, members, list_of_best_in_each_iter, "NN")

                """
                looks for the best model using svm
                """
                # df_pop, list_of_best_in_each_iter = iterate(pop_size, num_of_data_points, num_of_iter, dim,
                #                                             mutation_min_alpha, mutation_delta, df_train, df_valid,
                #                                             df_test, create_configuration_space_svm, predict_svm)
                #
                # write_results(df_pop, results_dir_str, members, list_of_best_in_each_iter, 'SVM')
                # results_df = results_df.append(pd.DataFrame([['svm', i, str(members), df_pop['results'].max(),
                #                                               df_pop['results_family'].max()]],
                #                                             columns=results_df.columns), ignore_index=True)
          except:
            print("problem in i = ", str(i))
    results_df.to_excel(str(results_dir_str + '/results from all.xlsx'))


