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
        ## augemntation
        if phase == "train":
            depth_img_numpy, angle = self.randomlySlidePixels(depth_img_numpy)
            acc_numpy = self.rotateVector(acc_numpy, angle)
        ## img: numpy -> tensor
        depth_img_numpy = depth_img_numpy.astype(np.float32)
        depth_img_numpy = self.addFlatnessCh(depth_img_numpy)
        depth_img_numpy[0] = np.where(depth_img_numpy[0] > 0, 1.0/depth_img_numpy[0], depth_img_numpy[0])
        depth_img_tensor = torch.from_numpy(depth_img_numpy)
        ## acc: numpy -> tensor
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return depth_img_tensor, acc_tensor

    def addFlatnessCh(self, depth_img_numpy):
        roll_plus = np.roll(depth_img_numpy, 1, axis=1)
        roll_plus = np.where(roll_plus < 0, 0, roll_plus)
        roll_minus = np.roll(depth_img_numpy, -1, axis=1)
        roll_minus = np.where(roll_minus < 0, 0, roll_minus)
        flatness_img_numpy = np.where(depth_img_numpy > 0.0, np.abs(2 * depth_img_numpy - roll_plus - roll_minus), depth_img_numpy)
        depth_img_numpy = np.stack([depth_img_numpy, flatness_img_numpy], 0)
        return depth_img_numpy

    def randomlySlidePixels(self, depth_img_numpy):
        ## slide: right -> left (rotate: CCW along Z-axis)
        slide_pixel = random.randint(0, depth_img_numpy.shape[1] - 1)
        slide_rad = self.anglePiToPi(2 * math.pi / depth_img_numpy.shape[1] * slide_pixel)
        slid_depth_img_numpy = np.roll(depth_img_numpy, -slide_pixel, axis=1)
        # print("slide_pixel = ", slide_pixel)
        # print("slide_rad/math.pi*180.0 = ", slide_rad/math.pi*180.0)
        return slid_depth_img_numpy, slide_rad

    def rotateVector(self, acc_numpy, angle):
        rot = np.array([
    	    [math.cos(-angle), -math.sin(-angle), 0.0],
    	    [math.sin(-angle), math.cos(-angle), 0.0],
    	    [0.0, 0.0, 1.0]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

    def anglePiToPi(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

##### test #####
# ## depth image
# depth_img_path = "../../../dataset_image_to_gravity/AirSim/lidar/example.npy"
# depth_img_numpy = np.load(depth_img_path)
# print("depth_img_numpy = ", depth_img_numpy)
# print("depth_img_numpy.shape = ", depth_img_numpy.shape)
# ## label
# acc_list = [1, 0, 0]
# acc_numpy = np.array(acc_list)
# print("acc_numpy = ", acc_numpy)
# ## transform
# transform = DataTransform()
# depth_img_trans, acc_trans = transform(depth_img_numpy, acc_numpy, phase="train")
# print("depth_img_trans[0] = ", depth_img_trans[0])
# print("depth_img_trans[1] = ", depth_img_trans[1])
# print("depth_img_trans.size() = ", depth_img_trans.size())
# print("acc_trans = ", acc_trans)
# ## tensor -> numpy
# depth_img_trans_numpy = depth_img_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
# print("depth_img_trans_numpy.shape = ", depth_img_trans_numpy.shape)
# ## save
# img_pil = Image.fromarray(np.uint8(255*depth_img_trans_numpy[:,:,0]/np.max(depth_img_trans_numpy[:,:,0])))
# save_path = "../../save/transform.jpg"
# img_pil.save(save_path)
# print("saved: ", save_path)
# ## imshow
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.imshow(depth_img_numpy)
# plt.subplot(3, 1, 2)
# plt.imshow(depth_img_trans_numpy[:,:,0])
# plt.subplot(3, 1, 3)
# plt.imshow(depth_img_trans_numpy[:,:,1])
# plt.show()
