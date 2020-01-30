from __future__ import absolute_import, division, print_function
import numpy as np



import os.path


def write_points_to_file(fun, lbounds, ubounds, problem_num):
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim = len(lbounds)
    if dim != 10:  # choose only dim 10
        return lbounds + (ubounds - lbounds) * np.random.rand(dim)
    if str(problem_num)[0] == "0":
        problem_num = str(problem_num)[1:]
    file_name = "test"
    save_path = r'C:\Users\ido\PycharmProjects\fuctions-learning\points'

    name_of_file = str(file_name)
    completeName = os.path.join(save_path, name_of_file+".txt")
    out_file = open(str(completeName), "a")

    num_of_points = 30  # num of points in every feature vector
    for i in range(2):  # num of samples for every instance
        X = lbounds + (ubounds - lbounds) * np.random.rand(num_of_points, dim)
        F = [fun(x) for x in X]
        str_out = ""
        for point in range(num_of_points):
            for coardinate in range(dim):
                str_out += str(float(X[point][coardinate])) + ", "
            str_out += str(F[point]) + ", "
        str_out += str(problem_num) + ".0\n"
        out_file.write(str(str_out))

    return X[0]
