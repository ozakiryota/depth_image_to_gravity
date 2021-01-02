import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import time

import torch
from torchvision import models
import torch.nn as nn

import sys
sys.path.append('../')
from common import inference_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod

def main():
    ## hyperparameters
    list_rootpath = ["../../../dataset_image_to_gravity/AirSim/lidar/val"]
    csv_name = "imu_lidar.csv"
    batch_size = 10
    weights_path = "../../weights/regression.pth"
    ## dataset
    dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(list_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(),
        phase="val"
    )
    ## network
    net = network_mod.Network(dim_fc_out=3)
    ## criterion
    criterion = nn.MSELoss()
    ## infer
    inference = inference_mod.Inference(
        dataset,
        net, weights_path, criterion,
        batch_size
    )
    inference.infer()

if __name__ == '__main__':
    main()
