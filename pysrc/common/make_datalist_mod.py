import csv
import os

def makeDataList(list_rootpath, csv_name):
    data_list = []
    for rootpath in list_rootpath:
        csv_path = os.path.join(rootpath, csv_name)
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row[3] = os.path.join(rootpath, row[3])
                data_list.append(row)
    return data_list

##### test #####
# list_rootpath = ["../../../dataset_image_to_gravity/AirSim/lidar/train"]
# csv_name = "imu_lidar.csv"
# train_list = makeDataList(list_rootpath, csv_name)
# # print(train_list)
# print("example0: ", train_list[0][:3], train_list[0][3:])
# print("example1: ", train_list[1][:3], train_list[1][3:])
