# f1 = open("./points/dim=10_10000_points_30_every_time.txt", 'r')
# f2 = open("./points/test.txt", 'r')
f2 = open("./points/dim=10_10000_points_30_every_time_2for_each.txt", 'r')
counter = 0
file_num = 1
f_out = open(str("./points/1000 points in file/dim=10_10000_points_30_every_time_split.txt" + str(file_num)), 'w')
line_counter = 0
lines_in_file = 5000


# for line in f1:
#     print(line)
#     line_counter += 1
#     if line_counter % 1000 == 0:
#         print(line_counter)
#     if counter < 1000:
#         f_out.write(line)
#         counter += 1
#     else:
#         f_out.close()
#         file_num += 1
#         f_out = open(str("./points/1000 points in file/dim=10_10000_points_30_every_time_split.txt" + str(file_num)), 'w')
#         f_out.write(line)
#         counter = 1
#
for line in f2:
    if counter < lines_in_file:
        f_out.write(line)
        counter += 1
    else:
        f_out.close()
        file_num += 1
        f_out = open(str("./points/1000 points in file/dim=10_10000_points_30_every_time_split.txt" + str(file_num)), 'w')
        f_out.write(line)
        counter = 1

f_out.close()
# f1.close()
f2.close()


# for line in f2:
#     line_counter += 1
#     length = len(line.split())
# print(line_counter)
# print(length)
