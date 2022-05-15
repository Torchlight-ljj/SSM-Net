# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
 
import io
import requests
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import os
from model.ssmnet import SSM_Net

class Cam():
    def __init__(self, Net, weight_path, input_size, cam_layer_name, cuda_flag = True):
        super(Cam, self).__init__()
        self.cuda_flag = cuda_flag
        if self.cuda_flag:
            self.Net = Net.cuda()
        else:
            self.Net = Net
        try:
            self.Net.load_state_dict(torch.load(weight_path))
        except Exception as e:
            raise e
        self.input_size = input_size
        self.cam_layer_name = cam_layer_name
        self.fmap_block = list()
        self.grad_block = list()
        self.ori_shape = None

    def get_cam(self, img_path, out_dir):
        img_pil = Image.open(img_path)
        self.ori_shape = img_pil.size
        ori = transforms.ToTensor()(img_pil)
        ori = transforms.Resize((self.input_size, self.input_size))(ori)
        if self.cuda_flag:
            img_variable = ori.unsqueeze(0).cuda()
        else:
            img_variable = ori.unsqueeze(0)
        #model
        self.Net.eval()
        classes = list(key for key in range(4))
        self.Net.context_path.layer4.register_forward_hook(self.farward_hook)
        self.Net.context_path.layer4.register_backward_hook(self.backward_hook)

        # forward
        seg, output, features = self.Net(img_variable)
        idx = np.argmax(output.cpu().data.numpy(),axis=1)[0]
        print("predict: {}".format(classes[idx]))

        # backward
        self.Net.zero_grad()
        class_loss = output[0,idx]
        class_loss.backward()

        # generate cam
        grads_val = self.grad_block[0].cpu().data.numpy().squeeze()
        fmap = self.fmap_block[0].cpu().data.numpy().squeeze()
        self.cam_show_img(img_path, fmap, grads_val, out_dir)

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def cam_show_img(self, img_path, feature_map, grads, out_dir):
        img = cv2.imread(img_path)
        # H, W, _ = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		
        grads = grads.reshape([grads.shape[0],-1])					
        weights = np.mean(grads, axis=1)							
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]							
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (512, 512))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img = cv2.resize(img, (512, 512))
        path_cam_img = out_dir
        cv2.imwrite(path_cam_img, heatmap)
Net = SSM_Net(n_channels=3, n_classes=2, num_classes=4)

img_list = [
'./data/ori/1/孙春艳-肿块/孙春艳_女_47岁_06419483_检查号_EUS49871-0120201221152831005.jpg',
'./data/ori/1/孙玉山-肿块/孙玉山_男_58岁_03344775_检查号_EUS17379-0120201221103154011.jpg',
'./data/ori/1/王小其/王小其_男_64岁_05995642_检查号_EUS46266-0120200918154813026.jpg',
'./data/ori/2/蒋红新-肿块型慢胰/蒋红新_男_41岁_QG00018051_检查号_EUS44869-0120201218153417067.jpg',
'./data/ori/2/舒松珠/舒松珠_女_65岁_04211906_检查号_EUS23517-0120200918161131050.jpg',
'./data/ori/2/张小平-胰腺低回声-肿块型慢胰/张小平_男_37岁_E213493_检查号_EUS13315-0220201216155807017.jpg',
'./data/ori/3/陈明/陈明_男_41岁_05532207_检查号_EUS40210-0120200921161614002.jpg',
'./data/ori/3/史锁年/史锁年_男_63岁_05651350_检查号_EUS42247-0120200921152424094.jpg',
'./data/ori/3/谢杏英/谢杏英_女_63岁_E192584_检查号_EUS14213-0320201211155430004.jpg',
]
for img in img_list:
    CAM = Cam(Net,'./SSM-Net/save/6.pth',512,'refine',True)
    CAM.get_cam(img, './grad_cam/' + img.split('/')[-1])