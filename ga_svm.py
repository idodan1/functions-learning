#!/usr/bin/env python3

from create_svm import predict
from functions import find_index
import numpy as np
import random


def create_pop(pop_size, configuration_space, len_data):
    """
    creates a list of dictionaries, each dict is a configuration representing an svm model.
    """
    pop = []
    for i in range(pop_size):
        member = {}
        for key in configuration_space.keys():
            key_type = type(configuration_space[key][0])
            if key_type is str:
                member[key] = configuration_space[key][np.random.randint(0, len(configuration_space[key]))]
            elif key_type is float:
                limits = configuration_space[key]
                member[key] = float('%.2f' % np.random.uniform(limits[0], limits[1]))
            elif key_type is int:
                limits = configuration_space[key]
                member[key] = int(np.random.randint(limits[0], limits[1]))
            else:
                print("illegal type")
        # a random order of points to create the model from
        member["sample"] = random.sample(range(0, len_data), len_data)
        pop.append(dict(member))
    return pop


def recombinant(x, y, f_x, f_y):
    # gives the better configuration more weight
    f_x = f_x ** 2
    f_y = f_y ** 2
    sum_square = f_x + f_y
    son = {}
    for key in x.keys():
        key_type = type(x[key])

        if key_type is str:
            alpha = np.random.uniform(0, 1)
            if alpha > f_x / sum_square:
                son[key] = y[key]
            else:
                son[key] = x[key]

        elif key_type is float:
            # does a mean of values according to fitness weight
            # max is for avoiding zero values which creates problems later
            son[key] = max(0.000001, float('%.2f' % (x[key] * f_x / sum_square + y[key] * f_y / sum_square)))

        elif key_type is int:
            son[key] = int((x[key] * f_x / sum_square + y[key] * f_y / sum_square))
    son["sample"] = add_sample(x["sample"], y["sample"])
    return son


def add_sample(sample1, sample2):
    """
    creates a new points sample from parents sample lists
    """
    sample_list = []
    used_list = [True] * len(sample1)
    now_1 = True
    i1 = 0
    i2 = 0
    while len(sample_list) < len(sample1):
        if now_1:
            if used_list[sample1[i1]]:
                used_list[sample1[i1]] = False
                sample_list.append(sample1[i1])
            now_1 = False
            i1 += 1
        else:
            if used_list[sample1[i2]]:
                used_list[sample2[i2]] = False
                sample_list.append(sample2[i2])
            now_1 = True
            i2 += 1
    return sample_list


def mutate(x, configuration_space, min_alpha, delta, len_data):
    mutation = {}
    for key in x.keys():
        key_type = type(x[key])

        if key_type is str:
            alpha = np.random.uniform(0, 1)
            if alpha < min_alpha:
                mutation[key] = configuration_space[key][np.random.randint(0, len(configuration_space[key]))]
            else:
                mutation[key] = x[key]

        elif key_type is float:
            factor = np.random.uniform(-1, 1)
            range_of_key = configuration_space[key][1] - configuration_space[key][0]
            # max is for avoiding zero values which creates problems later
            # min is for avoiding getting a value to high
            mutation[key] = max(0.000001, float('%.2f' % max(min((x[key] + factor * range_of_key * delta),
                                                                 configuration_space[key][1]),
                                                             configuration_space[key][0])))

        elif key_type is int:
            factor = np.random.randint(-2, 2)
            mutation[key] = int(max(min((x[key] + factor), configuration_space[key][1]), configuration_space[key][0]))

    num_of_points = int(len_data * mutation["percent_of_points"])
    mutation["sample"] = x["sample"]
    if num_of_points < len_data:
      for i in range(num_of_points):
          alpha = np.random.uniform(0, 1)
          if alpha > 0.9:
              index = np.random.randint(num_of_points, len_data)
              mutation["sample"][i], mutation["sample"][index] = mutation["sample"][index], mutation["sample"][i]
    return mutation


def create_configuration_space(num_of_data_points):
    configuration_space = dict()
    configuration_space["kernel"] = ["rbf", "sigmoid"]
    configuration_space["C"] = [0.001, 1000.0]
    configuration_space["shrinking"] = ["true", "false"]
    configuration_space["degree"] = [1, 5]
    configuration_space["coef0"] = [0.0, 10.0]
    configuration_space["gamma"] = ["auto", "value"]
    configuration_space["gamma_value"] = [0.0001, 8.0]  # only if gamma is value
    configuration_space["percent_of_points"] = [0.1, 1]
    configuration_space["num_of_points"] = [1, num_of_data_points]
    return configuration_space


def calc_pop(pop, dim, train_x, train_y, val_x, val_y):
    """
    calculates the identification percentage for every member in pop
    """
    results = []
    results_family = []
    i = 1
    for member in pop:
        i += 1
        res, res_family = predict(member, train_x, train_y, val_x, val_y, dim)
        results.append(res)
        results_family.append(res_family)
    return results, results_family


def create_cumsum(results):
    results_squared = [x ** 3 for x in results]
    # this way good results will stand out more
    results_sum = np.sum(results_squared)
    results_normalized = [float(results_squared[i]) / results_sum for i in range(len(results_squared))]
    results_cum_sum = np.cumsum(results_normalized)
    return results_cum_sum


def iterate(pop_size, num_of_data_points, num_of_iter, dim, mutation_min_alpha, mutation_delta,
            train_x, train_y, val_x, val_y, test_x, test_y):
    """
    returns a list of the best members of the final population with their test values and their configuration.
    """
    list_of_best_in_each_iter = []
    configuration_space = create_configuration_space(num_of_data_points)
    pop = create_pop(pop_size, configuration_space, len(train_y))
    for i in range(num_of_iter):
        if i % 10 == 0:
            print("\titer = " + str(i))
        results, results_family = calc_pop(pop, dim, train_x, train_y, val_x, val_y)
        results_cum_sum = create_cumsum(results)

        new_pop = []
        for j in range(pop_size):
            alpha = np.random.uniform(0, 1, 2)
            index1, index2 = find_index(results_cum_sum, alpha[0]), find_index(results_cum_sum, alpha[1])
            son = recombinant(pop[index1], pop[index2], results[index1], results[index2])
            son = mutate(son, configuration_space, mutation_min_alpha, mutation_delta, len(train_y))
            new_pop.append(son)

        new_results, new_results_family = calc_pop(new_pop, dim, train_x, train_y, val_x, val_y)
        # sort
        idx = np.argsort(results)
        pop = np.array(pop)[idx]
        results = np.array(results)[idx]
        # leaves only the 20% best members and add them to the new pop
        best_members_percent = int(len(pop)*0.2)
        pop = pop[len(pop) - best_members_percent:]
        pop = np.append(pop, new_pop)
        results = results[len(results) - best_members_percent:]
        results = np.append(results, new_results)

        # sorts again with the new pop and leaves the best pop_size from pop
        idx = np.argsort(results)
        pop = np.array(pop)[idx]
        results = np.array(results)[idx]
        pop = pop[best_members_percent:]
        results = results[best_members_percent:]
        pop = pop.tolist()
        list_of_best_in_each_iter.append(results[-1])
    test_results, test_results_family = calc_pop(pop, dim, train_x, train_y, test_x, test_y)
    # sort
    idx = np.argsort(test_results)
    pop = np.array(pop)[idx]
    test_results = np.array(test_results)[idx]
    return pop, test_results, test_results_family, list_of_best_in_each_iter




