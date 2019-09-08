#!/usr/bin/env python3

from ga_svm import iterate
from functions import get_data_from_file, create_data_for_model
from write_results import create_date_str, write_results, make_parameter_string
import os
import random
import datetime
import pandas as pd

if __name__ == "__main__":
    # points_file_name1 = "./points/dim=10_10000_points_30_every_time.txt"
    # points_file_name1 = "./points/dim=10_10000_points_30_every_time_split(1000).txt"
    # points_file_name2 = "./points/dim=10_10000_points_30_every_time_2for_each.txt"
    results_dir_str = "./results_svm_dim10/" + create_date_str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(results_dir_str)
    dir_list = os.listdir('./points/5000 points in file')
    df = get_data_from_file([str('./points/5000 points in file/' + file_name) for file_name in dir_list])
    dim = 10
    num_of_data_points = int(df.shape[1]/(dim+1))
    pop_size = 5
    num_of_iter = 2
    train_ratio = 0.01
    validation_ratio = 0.6
    test_ratio = 1
    mutation_min_alpha = 0.2
    mutation_delta = 0.1
    parameter_string = make_parameter_string(pop_size, num_of_iter, train_ratio, validation_ratio, test_ratio)
    f = open(results_dir_str + "/parameters", "w")
    f.write(parameter_string)
    f.close()
    results_df = pd.DataFrame(columns=['group size', 'members', 'best result', 'best result family'])
    
    for k in range(1):
        for i in range(3, 4):
            print("group size equals " + str(i))
            for j in range(1):
                members = random.sample(range(1, 25), i)
                df_train, df_valid, df_test = create_data_for_model(train_ratio, validation_ratio,
                                                                    test_ratio, members, df)
                len_data = df.shape[0]  # num of rows in df

                df_pop, list_of_best_in_each_iter = iterate(pop_size, num_of_data_points, num_of_iter, dim,
                                                            mutation_min_alpha,mutation_delta,df_train, df_valid,
                                                            df_test)

                print(df_pop['results'])

                write_results(df_pop, results_dir_str, members, list_of_best_in_each_iter)
                results_df = results_df.append(pd.DataFrame([[i, str(members), df_pop['results'].max(),
                                                              df_pop['results_family'].max()]],
                                                            columns=results_df.columns), ignore_index=True)

    results_df.to_excel(str(results_dir_str + '/results from all.xlsx'))


