import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil


def show_parameters(dir_name):
    df = pd.read_excel(str(dir_name + '/parameters.xlsx'))
    print(df.to_string())
    
    
def delete_empty_dirs(dir_name, dir_list):
    for d in dir_list:
        n = dir_name + '/' + d
        if len(os.listdir(n)) < 2:
            shutil.rmtree(n)


def analyze_points_distribution(dir_name):
    dir_list = os.listdir(dir_name)
    analyzing = True
    while analyzing:
        print('you have ' + str(len(dir_list)) + ' directories in results folder')
        for i in range(1, len(dir_list)+1):
            print('folder number ' + str(i))
            d = dir_name + '/' + dir_list[i-1]
            print(d)
            print(show_parameters(d))
            res_df = pd.read_excel(str(d + '/results from all.xlsx'))
            print('best results in folder = ' + str(res_df[0:1]['best result'].values))
            print('best results for family in folder = ' + str(res_df[0:1]['best result family'].values))
            print("\n")
        dir_num = int(input('which dir would you like to examine? (insert a number) '))
        dir_to_examine = str(dir_name + '/' + dir_list[dir_num - 1])
        df = pd.read_excel(str(dir_to_examine + '/wrong predictions net.xlsx'))
        plt.scatter(df.index.tolist(), df['right_percent'].values)
        plt.show()
        plt.clf()

        s = input("do you want to see point distribution(y/n)? ")
        if s == 'y':
            in_points = True
            while in_points:
                p = int(input('choose a number between 1 to 24: '))
                plt.scatter(df.index.tolist(), df[p:p+1][[i for i in range(1, 25)]].values[0])
                plt.show()
                plt.clf()
                answer = input('want to see other points?(y/n)')
                if answer != 'y':
                    in_points = False

        answer = input('go on analyzing?(y/n)')
        if answer != 'y':
            analyzing = False


dir_name_res = './results_dim10'
analyze_points_distribution(dir_name_res)

