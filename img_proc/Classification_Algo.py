import cv2 as cv
import numpy as np
from random import randint as ri
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from img_proc.Select_Algo import pre_proc, segmentBackground, selectSign

def fillImgWithZero(src_img, fill_size):
    (F_H, F_W) = fill_size
    (I_H, I_W) = src_img.shape

    fill_img = np.zeros((F_H, F_W), dtype=np.uint8)
    fill_img[0:I_H, 0:I_W] = src_img

    return fill_img

def scaleImg(src_img, max_l):
    (H, W) = src_img.shape
    L = max(H, W)
    ratio = max_l / L
    resize_h = int(H * ratio)
    resize_w = int(W * ratio)

    resize_img = cv.resize(src_img, (resize_h, resize_w), interpolation=cv.INTER_AREA)
    return resize_img

def getImgFromFile(label):
    bgr_img = cv.imread(f"datasets/train/{label}.bmp")
    seg_img = cv.imread(f"datasets/train_label/{label}.bmp", cv.IMREAD_GRAYSCALE)
    return (bgr_img, seg_img)

class ShapeNet(nn.Module):
    def __init__(self, class_num) -> None:
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.class_net = nn.Sequential(
            nn.Linear(3 * 16 * 16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.class_net(x)
        return x

def hasRectange(img, i_size=64, threshold=0.6):
    if (img is None):
        return False

    L = i_size
    img = scaleImg(img, i_size)
    img = fillImgWithZero(img, (i_size, i_size))
    cv.imwrite(f"save/temp/{ri(0, 9)}{ri(0, 9)}{ri(0, 9)}{ri(0, 9)}{ri(0, 9)}.jpg", img)
    L_MIN = int(L * threshold)
    lines = cv.HoughLines(img, 1, np.pi / 180, 50, None, L_MIN, 10)
    if (lines is None):
        return False
    elif (len(lines) >= 4):
        return True
    else:
        return False

def hasCircle(img, i_size=64, threshold=0.8):
    if (img is None):
        return False

    L = i_size 
    img = scaleImg(img, i_size)
    img = fillImgWithZero(img, (i_size, i_size))
    cv.imwrite(f"save/temp/{ri(0, 9)}{ri(0, 9)}{ri(0, 9)}{ri(0, 9)}{ri(0, 9)}.jpg", img)
    L_MIN = int(L/2 * threshold)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=20, minRadius=L_MIN, maxRadius=L)
    if (circles is None):
        return False
    else:
        return True

def classification(seg_img, net, dev):
    CIRCLE = 0
    OTHER = 1
    RECTANGE = 2
    img_trans = torchvision.transforms.ToTensor()
    label_list = []
    seg_img = pre_proc(seg_img)
    seg_img = segmentBackground(seg_img)
    (x1, x2, y1, y2) = selectSign(seg_img)
    for (w1, w2, h1, h2) in zip(x1, x2, y1, y2):
        gray_seg_img = np.uint8(seg_img[h1:h2, w1:w2])
        img = scaleImg(gray_seg_img, 64)
        img = fillImgWithZero(img, (64, 64))
        img = img_trans(img).to(dev).unsqueeze(0)
        label = int(torch.argmax(net(img)[0].cpu()).detach().numpy())
        label_list.append(int(label))
        #canny_seg_img = cv.Canny(gray_seg_img, 50, 150)
        #if hasCircle(gray_seg_img):
            #label_list.append(CIRCLE)
        #elif hasRectange(canny_seg_img):
            #label_list.append(RECTANGE)
        #else:
            #label_list.append(OTHER)

    gray_img = seg_img
    canny_img = cv.Canny(gray_img, 50, 150)
    return (x1, x2, y1, y2, label_list, gray_img, canny_img)
