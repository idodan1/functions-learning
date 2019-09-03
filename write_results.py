#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import pickle
import pandas as pd


# def write_results(pop, dir_str,  test_results, members, list_of_best, results_df):
def write_results(pop, dir_str,  test_results, members, list_of_best):

    # save results in a folder. father_folder_name == group_size. if not exist, create it.
    dir_str = dir_str + "/group size = " + str(len(members))
    if not os.path.exists(dir_str):
        os.mkdir(dir_str)
    # create a folder for the specific group of functions we are testing
    dir_str = dir_str + "/" + str(members)
    os.mkdir(dir_str)



    # save graph of best member in each iter

    try:
        df = pd.DataFrame(list_of_best, columns=["right prediction", "same family"])
        df.to_csv(str(dir_str + "/list_of_best.csv"))
        plt.scatter(range(1, len(list_of_best) + 1), list_of_best)
        plt.xlabel("iteration")
        plt.ylabel("validation percent")
        plt.savefig(str(dir_str + "/graph.png"))
        plt.clf()

    except:
        try:
            with open(dir_str + "/graph", "w") as output:
                pickle.dump(str(list_of_best), output, pickle.HIGHEST_PROTOCOL)
        except:
            f = open(dir_str + "/graph", "w")
            f.write(str(list_of_best))

    # f_out_points = open(dir_str + "/points_sample", "w")
    members_str = "members = " + str(members) + "\n"
    # f_out_points.write(members_str)
    f_out_configuration = open(dir_str + "/models_configuration", "w")
    f_out_configuration.write(members_str)

    columns = list(pop[0].keys())
    columns.remove('sample')
    columns.append('test results')
    df_pop = pd.DataFrame(columns=columns)

    for i in range(len(test_results)-1, -1, -1):

        # sample = str(pop[i]["sample"])
        # f_out_points.write(str(test_results[i]))
        # f_out_points.write("\n")
        # f_out_points.write(sample)
        # f_out_points.write("\n\n")
        del pop[i]["sample"]
        values = list(pop[i].values())
        values.append(test_results[i])
        df_pop = df_pop.append(pd.Series(values, index=columns), ignore_index=True)

        f_out_configuration.write(str(test_results[i]))

        f_out_configuration.write("\n")
        print(pop[i])
        f_out_configuration.write(str(pop[i]))

        f_out_configuration.write("\n\n")
    print(df_pop)
    f_out_configuration.close()

    # f_out_points.close()





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



















