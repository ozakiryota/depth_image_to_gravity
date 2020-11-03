from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class OriginalNet(nn.Module):
    def __init__(self):
        super(OriginalNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(57856, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, 3)
        )

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
# import data_transform_model
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
# transform = data_transform_model.DataTransform()
# depth_img_trans, _ = transform(depth_img_numpy, acc_numpy)
# ## network
# net = OriginalNet()
# print(net)
# list_cnn_param_value, list_fc_param_value = net.getParamValueList()
# ## prediction
# inputs = depth_img_trans.unsqueeze_(0)
# print("inputs.size() = ", inputs.size())
# outputs = net(inputs)
# print("outputs.size() = ", outputs.size())
