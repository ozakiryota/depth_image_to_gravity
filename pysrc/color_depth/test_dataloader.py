import numpy as np
import matplotlib.pyplot as plt

import torch

import make_datalist_mod
import data_transform_mod
import dataset_mod

def show_inputs(inputs_color, inputs_depth):
    h = 10
    w = 2
    plt.figure()
    for i, (img_color_tensor, img_depth_tensor) in enumerate(zip(inputs_color, inputs_depth)):
        if i < h*w:
            ## color
            img_color_numpy = img_color_tensor.numpy().transpose((1, 2, 0))
            img_color_numpy = np.clip(img_color_numpy, 0, 1)
            plt.subplot(h, w, w*i + 1)
            plt.imshow(img_color_numpy)
            ## depth
            img_depth_numpy = img_depth_tensor.numpy().transpose((1, 2, 0))
            img_depth_numpy = img_depth_numpy.squeeze(2)
            plt.subplot(h, w, w*i + 2)
            plt.imshow(img_depth_numpy)
    plt.show()

## list
train_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar1cam/train"
val_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar1cam/val"
csv_name = "imu_lidar_camera.csv"
train_list = make_datalist_mod.makeDataList(train_rootpath, csv_name)
val_list = make_datalist_mod.makeDataList(val_rootpath, csv_name)

## trans param
resize = 224
mean = ([0.5, 0.5, 0.5])
std = ([0.5, 0.5, 0.5])

## dataset
train_dataset = dataset_mod.OriginalDataset(
    data_list=train_list,
    transform=data_transform_mod.DataTransform(resize, mean, std),
    phase="train"
)
val_dataset = dataset_mod.OriginalDataset(
    data_list=val_list,
    transform=data_transform_mod.DataTransform(resize, mean, std),
    phase="val"
)

# dataloader
batch_size = 10

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

batch_iterator = iter(dataloaders_dict["train"])
# batch_iterator = iter(dataloaders_dict["val"])
inputs_color, inputs_depth, labels = next(batch_iterator)

# print("inputs = ", inputs)
print("inputs_color.size() = ", inputs_color.size())
print("inputs_depth.size() = ", inputs_depth.size())
print("labels = ", labels)
print("labels[0] = ", labels[0])
print("labels.size() = ", labels.size())
show_inputs(inputs_color, inputs_depth)
