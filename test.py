#!/usr/bin/env python3

from ga import iterate
from functions import get_data_from_file, create_data_for_model
from write_results import create_date_str, write_results, make_parameter_string
import os
import random
import datetime

if __name__ == "__main__":
    points_file_name = "./points/small_points_dim10"
    results_dir_str = "./results_svm_dim10/" + create_date_str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(results_dir_str)
    data_x, data_y = get_data_from_file(points_file_name)
    dim = 10
    num_of_data_points = len(data_x[0])/(dim+1)
    pop_size = 1000
    num_of_iter = 100
    train_ratio = 0.25
    validation_ratio = 0.4
    test_ratio = 0.8
    mutation_min_alpha = 0.2
    mutation_delta = 0.1
    parameter_string = make_parameter_string(pop_size, num_of_iter, train_ratio, validation_ratio, test_ratio)
    f = open(results_dir_str + "/parameters", "w")
    f.write(parameter_string)
    f.close()

    for i in range(2, 25):
        print("group size equals " + str(i))
        for j in range(10):
            members = random.sample(range(1, 25), i)
            train_x, train_y, validation_x, validation_y, test_x, test_y = \
                create_data_for_model(data_x, data_y, train_ratio, validation_ratio,
                                      test_ratio, members)
            len_data = len(train_x)

            pop, test_results, list_of_best_in_each_iter = iterate(pop_size, num_of_data_points, num_of_iter,
                                                                   dim, mutation_min_alpha, mutation_delta, train_x,
                                                                   train_y, validation_x, validation_y, test_x, test_y)

            write_results(pop, results_dir_str, test_results, members, list_of_best_in_each_iter)




