import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

import make_datalist_mod
import data_transform_mod
import dataset_mod

def save_examples(inputs, labels):
    num = 10
    fig, ax = plt.subplots()
    y_axis = ax.quiver(-0.5, 0, color='green', angles='xy', scale_units='xy', scale=1)
    z_axis =  ax.quiver(0, 0.5, color='blue', angles='xy', scale_units='xy', scale=1)
    for i, tensor in enumerate(inputs):
        if i < num:
            ## input
            img = tensor.numpy().transpose((1, 2, 0))
            img = img.squeeze(2)
            img_pil = Image.fromarray(np.uint8(255*img/np.max(img)))
            img_pil.save("../../save/example" + str(i) + "_input.jpg")
            ## label
            q = ax.quiver(labels[i][1], -labels[i][2], color='deepskyblue', angles='xy', scale_units='xy', scale=1)
            ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            plt.savefig("../../save/example" + str(i) + "_label.jpg")
            q.remove()
    y_axis.remove()
    z_axis.remove()

def show_examples(inputs, labels):
    h = 10
    w = 1
    # plt.figure()
    for i, tensor in enumerate(inputs):
        if i < h*w:
            img = tensor.numpy().transpose((1, 2, 0))
            img = img.squeeze(2)
            ## input
            ax_input = plt.subplot(h, 2*w, 2*i+1)
            # ax_input.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax_input.imshow(img)
            ## label
            ax_label = plt.subplot(h, 2*w, 2*i+2)
            ax_label.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax_label.set_xlim([-1, 1])
            ax_label.set_ylim([-1, 1])
            ax_label.quiver(labels[i][1], -labels[i][2], color='red', angles='xy', scale_units='xy', scale=1)
    plt.show()

## list
list_train_rootpath = ["../../../dataset_image_to_gravity/AirSim/lidar/train"]
list_val_rootpath = ["../../../dataset_image_to_gravity/AirSim/lidar/val"]
csv_name = "imu_lidar.csv"
train_list = make_datalist_mod.makeDataList(list_train_rootpath, csv_name)
val_list = make_datalist_mod.makeDataList(list_val_rootpath, csv_name)

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
save_examples(inputs, labels)
show_examples(inputs, labels)
