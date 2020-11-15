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
            depth_img_numpy, heading_rad = self.randomlySlidePixels(depth_img_numpy)
            acc_numpy = self.rotateVector(acc_numpy, -heading_rad)
        ## img: numpy -> tensor
        depth_img_numpy = depth_img_numpy.astype(np.float32)
        depth_img_numpy = np.where(depth_img_numpy > 0.0, 1.0/depth_img_numpy, depth_img_numpy)
        # depth_img_numpy = np.clip(depth_img_numpy, 0, 1)
        depth_img_tensor = torch.from_numpy(depth_img_numpy)
        depth_img_tensor = depth_img_tensor.unsqueeze_(0)
        ## acc: numpy -> tensor
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return depth_img_tensor, acc_tensor

    def randomlySlidePixels(self, depth_img_numpy):
        ## slide: left -> right (rotate: CCW along Z-axis)
        slide_pixel = random.randint(0, depth_img_numpy.shape[1] - 1)
        heading_rad = self.anglePiToPi(2 * math.pi / depth_img_numpy.shape[1] * slide_pixel)
        slid_depth_img_numpy = np.roll(depth_img_numpy, slide_pixel, axis=1)
        # print("slide_pixel = ", slide_pixel)
        # print("heading_rad/math.pi*180.0 = ", heading_rad/math.pi*180.0)
        return slid_depth_img_numpy, heading_rad

    def rotateVector(self, acc_numpy, angle):
        rot = np.array([
    	    [math.cos(angle), -math.sin(angle), 0.0],
    	    [math.sin(angle), math.cos(angle), 0.0],
    	    [0.0, 0.0, 1.0]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

    def anglePiToPi(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

##### test #####
## depth image
depth_img_path = "../../../dataset_image_to_gravity/AirSim/lidar/example.npy"
depth_img_numpy = np.load(depth_img_path)
print("depth_img_numpy = ", depth_img_numpy)
print("depth_img_numpy.shape = ", depth_img_numpy.shape)
## label
acc_list = [1, 0, 0]
acc_numpy = np.array(acc_list)
print("acc_numpy = ", acc_numpy)
## transform
transform = DataTransform()
depth_img_trans, acc_trans = transform(depth_img_numpy, acc_numpy, phase="train")
print("acc_trans = ", acc_trans)
## tensor -> numpy
depth_img_trans_numpy = depth_img_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
print("depth_img_trans_numpy.shape = ", depth_img_trans_numpy.shape)
## save
depth_img_trans_numpy = depth_img_trans_numpy.squeeze(2)
img_pil = Image.fromarray(np.uint8(255*depth_img_trans_numpy/np.max(depth_img_trans_numpy)))
save_path = "../../save/transform.jpg"
img_pil.save(save_path)
print("saved: ", save_path)
## imshow
plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(depth_img_numpy)
plt.subplot(2, 1, 2)
plt.imshow(depth_img_trans_numpy)
plt.show()
