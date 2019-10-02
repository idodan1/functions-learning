#!/usr/bin/env python3

from functions import find_index
import numpy as np
import pandas as pd


def create_pop(pop_size, configuration_space):
    """
    creates a list of dictionaries, each dict is a configuration representing an svm model.
    """
    df_pop = pd.DataFrame(columns=configuration_space.keys())
    for i in range(pop_size):
        mem = []
        print(configuration_space)
        for key in configuration_space.keys():
            key_type = type(configuration_space[key][0])
            if key_type is str:
                mem.append(configuration_space[key][np.random.randint(0, len(configuration_space[key]))])
            elif key_type is float:
                limits = configuration_space[key]
                mem.append(float('%.2f' % np.random.uniform(limits[0], limits[1])))
            elif key_type is int:
                limits = configuration_space[key]
                mem.append(int(np.random.randint(limits[0], limits[1])))
            else:
                print("illegal type")

        df_pop = df_pop.append(pd.DataFrame([mem], columns=configuration_space.keys()), ignore_index=True)
    return df_pop


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
    return son


def mutate(x, configuration_space, min_alpha, delta):
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
    #         min/max is for staying in configuration space boundaries

    return mutation


def create_cumsum(results):
    results_squared = [x ** 3 for x in results]
    # this way good results will stand out more
    results_sum = np.sum(results_squared)
    results_normalized = [float(results_squared[i]) / results_sum for i in range(len(results_squared))]
    results_cum_sum = np.cumsum(results_normalized)
    return results_cum_sum


def iterate(pop_size, num_of_data_points, num_of_iter, dim, mutation_min_alpha, mutation_delta,
            df_train, df_valid, df_test, create_configuration_space, predict, members, wrong_predictions_df):
    """
        returns a list of the best members of the final population with their test values and their configuration.
    """
    models_history_dict = {}
    list_of_best_in_each_iter = []
    configuration_space = create_configuration_space(num_of_data_points)
    df_pop = create_pop(pop_size, configuration_space)
    for i in range(num_of_iter):
        # if i % 10 == 0:
        print("\titer = " + str(i))
        # returns a df of models we've already tested with their results (pop_old) and a df of new
        # models (pop_new) that will be predicted
        df_pop_old, df_pop_new = find_in_history_dict(models_history_dict,
                                                      df_pop[[col for col in configuration_space.keys()]])
        # the for loop here is because we need to remove results and results_family but they are not there
        # in the first iter
        df_pop_new = predict(df_pop_new, df_train, df_valid, df_test, dim, members, in_training=True)
        df_pop = df_pop_old.append(df_pop_new, ignore_index=True)

        add_to_dict(models_history_dict, df_pop)
        results_cum_sum = create_cumsum(df_pop['results'].values)

        new_pop = pd.DataFrame(columns=configuration_space.keys())
        for j in range(pop_size):
            alpha = np.random.uniform(0, 1, 2)
            index1, index2 = find_index(results_cum_sum, alpha[0]), find_index(results_cum_sum, alpha[1])
            son = recombinant(dict(zip(df_pop.columns.tolist(), df_pop[index1:index1+1].drop(['results', 'results_family'], axis=1).values.tolist()[0])),
                              dict(zip(df_pop.columns.tolist(), df_pop[index2:index2+1].drop(['results', 'results_family'], axis=1).values.tolist()[0])),
                              df_pop[index1:index1+1]['results'].values,
                              df_pop[index2:index2+1]['results'].values)
            son = mutate(son, configuration_space, mutation_min_alpha, mutation_delta)
            # needed because dict values get mixed and we need them in the same order as configuration_space.keys
            list_for_append = [son[key] for key in configuration_space.keys()]
            new_pop = new_pop.append(pd.Series(list(list_for_append), index=configuration_space.keys()), ignore_index=True)

        new_pop_old, new_pop_new = find_in_history_dict(models_history_dict, new_pop)
        new_pop_new = predict(new_pop_new, df_train, df_valid, df_test, dim, members, in_training=True)
        new_pop = new_pop_old.append(new_pop_new, ignore_index=True)

        df_pop = df_pop.sort_values(by=['results'], ascending=False)
        # leaves only the 20% best members and add them to the new pop
        df_pop = df_pop[0:int(len(df_pop)*0.2)]
        df_pop = df_pop.append(new_pop, ignore_index=True)
        df_pop = df_pop.sort_values(by=['results'], ascending=False)
        df_pop = df_pop[0:pop_size]
        list_of_best_in_each_iter.append(df_pop['results'].max())

    df_pop = predict(df_pop.drop(['results', 'results_family'], axis=1), df_train, df_valid, df_test, dim, members,
                     in_training=False, wrong_predictions_df=wrong_predictions_df)
    df_pop = df_pop.sort_values(by=['results'], ascending=False)
    return df_pop, list_of_best_in_each_iter


def add_to_dict(models_history_dict, df_pop):
    columns = list(df_pop.columns)
    columns.remove('results')
    columns.remove('results_family')
    for index, row in df_pop.iterrows():
        res = [row['results'], row['results_family']]
        row = row.drop(['results', 'results_family'])
        models_history_dict[create_str(row, columns)] = res


def create_str(row, columns):
    s = ''
    for col in columns:
        s += str(row[col]) + ','
    return s


def find_in_history_dict(models_history_dict, df_pop):
    old_models = models_history_dict.keys()
    columns = list(df_pop.columns)
    df_pop_new = pd.DataFrame(columns=columns)
    df_pop_old = pd.DataFrame(columns=columns+['results', 'results_family'])
    for index, row in df_pop.iterrows():
        model_string = create_str(row, columns)
        if model_string in old_models:
            res = models_history_dict[create_str(row, columns)]
            print(res)
            row['results'] = res[0]
            row['results_family'] = res[1]
            df_pop_old = df_pop_old.append(row, ignore_index=True)
        else:
            df_pop_new = df_pop_new.append(row, ignore_index=True)
    return df_pop_old, df_pop_new


