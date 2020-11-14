from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod

def main():
    ## hyperparameters
    method_name = "regression"
    train_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar/train"
    val_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar/val"
    csv_name = "imu_lidar.csv"
    optimizer_name = "Adam"  #"SGD" or "Adam"
    lr_cnn = 1e-5
    lr_fc = 1e-4
    batch_size = 100
    num_epochs = 50
    ## dataset
    train_dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(train_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(),
        phase="train"
    )
    val_dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(val_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(),
        phase="val"
    )
    ## network
    net = network_mod.Network(dim_fc_out=3)
    ## criterion
    criterion = nn.MSELoss()
    ## train
    trainer = trainer_mod.Trainer(
        method_name,
        train_dataset, val_dataset,
        net, criterion,
        optimizer_name, lr_cnn, lr_fc,
        batch_size, num_epochs
    )
    trainer.train()

if __name__ == '__main__':
    main()
