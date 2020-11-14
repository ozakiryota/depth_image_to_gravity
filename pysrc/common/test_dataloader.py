import numpy as np
import matplotlib.pyplot as plt

import torch

import make_datalist_mod
import data_transform_mod
import dataset_mod

def show_inputs(inputs):
    h = 10
    w = 1
    plt.figure()
    for i, tensor in enumerate(inputs):
        if i < h*w:
            img = tensor.numpy().transpose((1, 2, 0))
            img = img.squeeze(2)
            plt.subplot(h, w, i+1)
            plt.imshow(img)
    plt.show()

## list
train_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar/train"
val_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar/val"
csv_name = "imu_lidar.csv"
train_list = make_datalist_mod.makeDataList(train_rootpath, csv_name)
val_list = make_datalist_mod.makeDataList(val_rootpath, csv_name)

## dataset
train_dataset = dataset_mod.OriginalDataset(
    data_list=train_list,
    transform=data_transform_mod.DataTransform(),
    phase="train"
)
val_dataset = dataset_mod.OriginalDataset(
    data_list=val_list,
    transform=data_transform_mod.DataTransform(),
    phase="val"
)

# dataloader
batch_size = 10

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

batch_iterator = iter(dataloaders_dict["train"])
# batch_iterator = iter(dataloaders_dict["val"])
inputs, labels = next(batch_iterator)

# print("inputs = ", inputs)
print("inputs.size() = ", inputs.size())
print("labels = ", labels)
print("labels[0] = ", labels[0])
print("labels.size() = ", labels.size())
show_inputs(inputs)
