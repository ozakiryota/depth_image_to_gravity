from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, use_pretrained=True):
        super(Network, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained)
        self.color_cnn = vgg.features

        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.depth_cnn = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=(3, 5), stride=1, padding=(1, 2)),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
        #     nn.Conv2d(64, 128, kernel_size=(3, 5), stride=1, padding=(1, 2)),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
        #     nn.Conv2d(128, 256, kernel_size=(3, 5), stride=1, padding=(1, 2)),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        # )

        num_fc_in_features = 512*(resize//32)*(resize//32) + 256*(32//8)*(1812//8)
        self.fc = nn.Sequential(
            nn.Linear(num_fc_in_features, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(18, 3)
        )
        self.initializeWeights()

    def initializeWeights(self):
        for m in self.depth_cnn.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def getParamValueList(self):
        list_colorcnn_param_value = []
        list_depthcnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "color_cnn" in param_name:
                # print("color_cnn: ", param_name)
                list_colorcnn_param_value.append(param_value)
            if "depth_cnn" in param_name:
                # print("depth_cnn: ", param_name)
                list_depthcnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_colorcnn_param_value: ",list_colorcnn_param_value)
        # print("list_depthcnn_param_value: ",list_depthcnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_colorcnn_param_value, list_depthcnn_param_value, list_fc_param_value

    def forward(self, inputs_color, inputs_depth):
        ## cnn
        features_color = self.color_cnn(inputs_color)
        features_depth = self.depth_cnn(inputs_depth)
        ## concat
        features_color = torch.flatten(features_color, 1)
        features_depth = torch.flatten(features_depth, 1)
        features = torch.cat((features_color, features_depth), dim=1)
        ## fc
        outputs = self.fc(features)
        l2norm = torch.norm(outputs[:, :3].clone(), p=2, dim=1, keepdim=True)
        outputs[:, :3] = torch.div(outputs[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return outputs

##### test #####
# import data_transform_mod
# ## color image
# color_img_path = "../../../dataset_image_to_gravity/AirSim/example/camera_0.jpg"
# color_img_pil = Image.open(color_img_path)
# ## depth image
# depth_img_path = "../../../dataset_image_to_gravity/AirSim/example/lidar.npy"
# depth_img_numpy = np.load(depth_img_path)
# ## label
# acc_list = [0, 0, 1]
# acc_numpy = np.array(acc_list)
# ## trans param
# resize = 224
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## transform
# transform = data_transform_mod.DataTransform(resize, mean, std)
# color_img_trans, depth_img_trans, _ = transform(color_img_pil, depth_img_numpy, acc_numpy, phase="train")
# ## network
# net = Network(resize, use_pretrained=True)
# print(net)
# list_colorcnn_param_value, list_depthcnn_param_value, list_fc_param_value = net.getParamValueList()
# print("len(list_colorcnn_param_value) = ", len(list_colorcnn_param_value))
# print("len(list_depthcnn_param_value) = ", len(list_depthcnn_param_value))
# print("len(list_fc_param_value) = ", len(list_fc_param_value))
# ## prediction
# inputs_color = color_img_trans.unsqueeze_(0)
# inputs_depth = depth_img_trans.unsqueeze_(0)
# print("inputs_color.size() = ", inputs_color.size())
# print("inputs_depth.size() = ", inputs_depth.size())
# outputs = net(inputs_color, inputs_depth)
# print("outputs.size() = ", outputs.size())
# print("outputs = ", outputs)
