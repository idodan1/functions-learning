#!/usr/bin/env python3

from ga import iterate
from functions import get_data_from_file, create_data_for_model
from create_net import create_configuration_space_net, predict_net
from create_net_conv import predict_net_conv, create_configuration_space_cnn
from create_svm import create_configuration_space_svm, predict_svm
from write_results import create_date_str, write_results, make_parameter_df
import os
import random
import datetime
import pandas as pd

if __name__ == "__main__":
    results_dir_str = "./results_dim10/" + create_date_str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(results_dir_str)
    dir_list = os.listdir('./points')
    num_of_files_for_data = 1
    df = get_data_from_file([str('./points/' + file_name) for file_name in dir_list], num_of_files_for_data)
    dim = 10
    num_of_data_points = int(df.shape[1]/(dim+1))
    pop_size = 2
    num_of_iter = 2
    train_ratio = 0.5
    validation_ratio = 0.7
    test_ratio = 1
    mutation_min_alpha = 0.2
    mutation_delta = 0.1

    parameter_df = make_parameter_df(pop_size, num_of_iter, df.shape[0], train_ratio, validation_ratio, test_ratio,
                                     num_of_files_for_data, num_of_data_points)
    parameter_df.to_excel(str(results_dir_str + '/parameters.xlsx'))

    results_df = pd.DataFrame(columns=['algorithm', 'group size', 'members', 'best result', 'best result family'])

    columns = ['right_prediction', 'wrong_prediction', 'right_percent'] + [i for i in range(1, 25)]
    wrong_predictions_df_net = pd.DataFrame(columns=columns, index=[str(i) for i in range(1, 25)])
    wrong_predictions_df_net.loc[:, :] = 0
    wrong_predictions_df_net_conv = pd.DataFrame(columns=columns, index=[str(i) for i in range(1, 25)])
    wrong_predictions_df_net_conv.loc[:, :] = 0
    wrong_predictions_df_svm = pd.DataFrame(columns=columns, index=[str(i) for i in range(1, 25)])
    wrong_predictions_df_svm.loc[:, :] = 0

    for k in range(1):
        for i in range(4, 5, 2):
            print("group size equals " + str(i))
            members = random.sample(range(1, 25), i)
            df_train, df_valid, df_test = create_data_for_model(train_ratio, validation_ratio,
                                                                test_ratio, members, df)
            len_data = df.shape[0]  # num of rows in df

            """
            # looks for the best model using NN
            # """
            # df_pop, list_of_best_in_each_iter = iterate(pop_size, num_of_data_points, num_of_iter, dim,
            #                                             mutation_min_alpha, mutation_delta, df_train, df_valid,
            #                                             df_test, create_configuration_space_net, predict_net,
            #                                             members, wrong_predictions_df_net)
            #
            # results_df = results_df.append(pd.DataFrame([['NN', i, str(members), df_pop['results'].max(),
            #                                               df_pop['results_family'].max()]],
            #                                             columns=results_df.columns), ignore_index=True)
            # write_results(df_pop, results_dir_str, members, list_of_best_in_each_iter, "NN")

            """
            looks for the best model using CNN
            """
            df_pop, list_of_best_in_each_iter = iterate(pop_size, num_of_data_points, num_of_iter, dim,
                                                        mutation_min_alpha, mutation_delta, df_train, df_valid,
                                                        df_test, create_configuration_space_cnn, predict_net_conv,
                                                        members, wrong_predictions_df_net)

            results_df = results_df.append(pd.DataFrame([['CNN', i, str(members), df_pop['results'].max(),
                                                          df_pop['results_family'].max()]],
                                                        columns=results_df.columns), ignore_index=True)
            print(results_df)
            # write_results(df_pop, results_dir_str, members, list_of_best_in_each_iter, "CNN")

            """
            looks for the best model using svm
            """
            # df_pop, list_of_best_in_each_iter = iterate(pop_size, num_of_data_points, num_of_iter, dim,
            #                                             mutation_min_alpha, mutation_delta,
            #                                             df_train[0:int(0.1*len(df_train))],
            #                                             df_valid[0:int(0.1*len(df_train))],
            #                                             df_test[0:int(0.1*len(df_train))],
            #                                             create_configuration_space_svm, predict_svm, members,
            #                                             wrong_predictions_df_svm)

            # write_results(df_pop, results_dir_str, members, list_of_best_in_each_iter, 'SVM')
            # results_df = results_df.append(pd.DataFrame([['svm', i, str(members), df_pop['results'].max(),
            #                                               df_pop['results_family'].max()]],
            #                                             columns=results_df.columns), ignore_index=True)

    # results_df.to_excel(str(results_dir_str + '/results from all.xlsx'))
    # wrong_predictions_df_net['right_percent'] = wrong_predictions_df_net['right_prediction'].values/\
    #                                         (wrong_predictions_df_net['right_prediction'].values
    #                                          + wrong_predictions_df_net['wrong_prediction'].values)
    # wrong_predictions_df_net.to_excel(str(results_dir_str + '/wrong predictions net.xlsx'))

    # results_df.to_excel(str(results_dir_str + '/results from all.xlsx'))
    # wrong_predictions_df_net_conv['right_percent'] = wrong_predictions_df_net_conv['right_prediction'].values / \
    #                                             (wrong_predictions_df_net_conv['right_prediction'].values
    #                                              + wrong_predictions_df_net_conv['wrong_prediction'].values)
    # wrong_predictions_df_net.to_excel(str(results_dir_str + '/wrong predictions net conv.xlsx'))

    # wrong_predictions_df_svm['right_percent'] = wrong_predictions_df_svm['right_prediction'].values / \
    #                                             (wrong_predictions_df_svm['right_prediction'].values
    #                                              + wrong_predictions_df_svm['wrong_prediction'].values)
    # wrong_predictions_df_svm.to_excel(str(results_dir_str + '/wrong predictions svm.xlsx'))


