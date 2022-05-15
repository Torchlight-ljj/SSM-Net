import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ListDataset(Dataset):
    def __init__(self, list_path,transform = None, image_size = 256,train=True, is_ours = False):
        self.labels = []
        self.batch_count = 0
        self.transform = transform
        self.train = train
        self.image_size = image_size
        self.is_ours = is_ours
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
    def __getitem__(self, index):
        # ---------
        #  Label
        # ---------
        temp_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = temp_path.replace("jpg","png").replace("ori","mask")
        classes = int(label_path.split('/')[3])
        label_size = None
        if classes == 0:
            label = torch.zeros((self.image_size,self.image_size))
            label_size = label.shape
            label = torch.Tensor(label).unsqueeze(0)
        else:
            if self.is_ours:
                label = (cv2.imread(label_path,0)/255)*1
            else:
                label = (cv2.imread(label_path,0)/255)*classes
            label_size = label.shape
            label = torch.Tensor(label).unsqueeze(0)
            label = transforms.Resize((self.image_size,self.image_size))(label)

        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # Extract image as PyTorch tensor
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            img = transforms.Resize((label_size[0],label_size[1]))(img)
            img = transforms.Resize((self.image_size,self.image_size))(img)
        return img, label, classes
    def __len__(self):
        return len(self.img_files)
