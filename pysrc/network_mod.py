from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=(3, 5), stride=1, padding=(1, 2)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
        #     nn.Conv2d(64, 128, kernel_size=(3, 5), stride=1, padding=(1, 2)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
        #     nn.Conv2d(128, 256, kernel_size=(3, 5), stride=1, padding=(1, 2)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        # )
        self.fc = nn.Sequential(
            nn.Linear(231424, 100),
            # nn.Linear(28672, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, 3)
        )
        self.initializeWeights()

    def initializeWeights(self):
        for m in self.cnn.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def getParamValueList(self):
        list_cnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                # print("cnn: ", param_name)
                list_cnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_cnn_param_value: ",list_cnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_cnn_param_value, list_fc_param_value

    def forward(self, x):
        # print("cnn-in", x.size())
        x = self.cnn(x)
        # print("cnn-out", x.size())
        x = torch.flatten(x, 1)
        # print("fc-in", x.size())
        x = self.fc(x)
        # print("fc-out", x.size())
        l2norm = torch.norm(x[:, :3].clone(), p=2, dim=1, keepdim=True)
        x[:, :3] = torch.div(x[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return x

##### test #####
# import data_transform_mod
# ## ref
# # vgg = models.vgg16(pretrained=False)
# # print(vgg)
# ## depth image
# depth_img_path = "../../dataset_image_to_gravity/AirSim/lidar/example.npy"
# depth_img_numpy = np.load(depth_img_path)
# ## label
# acc_list = [0, 0, 1]
# acc_numpy = np.array(acc_list)
# ## transform
# transform = data_transform_mod.DataTransform()
# depth_img_trans, _ = transform(depth_img_numpy, acc_numpy)
# ## network
# net = Network()
# print(net)
# list_cnn_param_value, list_fc_param_value = net.getParamValueList()
# ## prediction
# inputs = depth_img_trans.unsqueeze_(0)
# print("inputs.size() = ", inputs.size())
# outputs = net(inputs)
# print("outputs.size() = ", outputs.size())
