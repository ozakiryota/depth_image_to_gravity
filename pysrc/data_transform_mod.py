from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self):
        print("DataTransform.__init__")

    def __call__(self, depth_img_numpy, acc_numpy, phase="train"):
        ## img
        depth_img_numpy = depth_img_numpy.astype(np.float32)
        depth_img_tensor = torch.from_numpy(depth_img_numpy)
        depth_img_tensor = depth_img_tensor.unsqueeze_(0)
        ## acc
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return depth_img_tensor, acc_tensor

    def rotateVector(self, acc_numpy, angle):
        rot = np.array([
            [1, 0, 0],
            [0, math.cos(-angle), -math.sin(-angle)],
            [0, math.sin(-angle), math.cos(-angle)]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

##### test #####
# ## depth image
# depth_img_path = "../../dataset_image_to_gravity/AirSim/lidar/example.npy"
# depth_img_numpy = np.load(depth_img_path)
# print("depth_img_numpy = ", depth_img_numpy)
# print("depth_img_numpy.shape = ", depth_img_numpy.shape)
# ## label
# acc_list = [0, 0, 1]
# acc_numpy = np.array(acc_list)
# ## transform
# transform = DataTransform()
# depth_img_trans, acc_trans = transform(depth_img_numpy, acc_numpy, phase="train")
# print("acc_trans = ", acc_trans)
# ## tensor -> numpy
# depth_img_trans_numpy = depth_img_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
# print("depth_img_trans_numpy.shape = ", depth_img_trans_numpy.shape)
# ## save
# depth_img_trans_numpy = depth_img_trans_numpy.squeeze(2)
# img_pil = Image.fromarray(np.uint8(255*depth_img_trans_numpy/np.linalg.norm(depth_img_trans_numpy)))
# save_path = "../save/transform.jpg"
# img_pil.save(save_path)
# print("saved: ", save_path)
# ## imshow
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.imshow(depth_img_numpy)
# plt.subplot(2, 1, 2)
# plt.imshow(depth_img_trans_numpy)
# plt.show()
