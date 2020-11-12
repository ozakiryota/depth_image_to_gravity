import torch.utils.data as data
from PIL import Image
import numpy as np

import torch

class OriginalDataset(data.Dataset):
    def __init__(self, data_list, transform, phase):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ## divide list
        depth_img_path = self.data_list[index][3]
        color_img_path = self.data_list[index][4]
        acc_str_list = self.data_list[index][:3]
        acc_list = [float(num) for num in acc_str_list]
        ## to numpy
        depth_img_numpy = np.load(depth_img_path)
        color_img_pil = Image.open(color_img_path)
        acc_numpy = np.array(acc_list)
        ## tansform
        color_img_trans, depth_img_trans, acc_trans = self.transform(color_img_pil, depth_img_numpy, acc_numpy, phase=self.phase)
        return color_img_trans, depth_img_trans, acc_trans

##### test #####
# import make_datalist_mod
# import data_transform_mod
# ## list
# train_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar1cam/train"
# val_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar1cam/val"
# csv_name = "imu_lidar_camera.csv"
# train_list = make_datalist_mod.makeDataList(train_rootpath, csv_name)
# val_list = make_datalist_mod.makeDataList(val_rootpath, csv_name)
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## dataset
# train_dataset = OriginalDataset(
#     data_list=train_list,
#     transform=data_transform_mod.DataTransform(resize, mean, std),
#     phase="train"
# )
# val_dataset = OriginalDataset(
#     data_list=val_list,
#     transform=data_transform_mod.DataTransform(resize, mean, std),
#     phase="val"
# )
# ## print
# index = 0
# print("index", index, ": ", train_dataset.__getitem__(index)[0].size())   #color
# print("index", index, ": ", train_dataset.__getitem__(index)[1].size())   #depth
# print("index", index, ": ", train_dataset.__getitem__(index)[2])   #label
