import numpy as np
import cv2 as cv
import torch
import torchvision

def matrix_torch2opencv(torch_tensor):
    ''' Shape: (C, H, W) -> (H, W) '''
    cv_array = np.array(torch_tensor.permute(1, 2, 0).cpu())
    cv_array = np.utin8(cv_array * 255)
    (H, W, _) = cv_array.shape
    return cv_array.reshape(H, W)

def pre_proc(img):
    #i = cv.GaussianBlur(img, (3, 3), 0)
    i = img
    p1 = i >= 128
    p2 = i < 128
    i[p1] = 255
    i[p2] = 0
    i = cv.erode(i, (3, 3))
    i = cv.dilate(i, (3, 3))
    i = cv.dilate(i, (3, 3))
    i = cv.erode(i, (3, 3))
    i[p1] = 0
    i[p2] = 255

    return i


def segmentBackground(img):
    ''' Get connected components and select the component which has the most points at image's edge as background '''
    (H, W) = img.shape
    (segment_num, segment_label) = cv.connectedComponents(img, connectivity=4)
    background_label = 0
    background_edge_num = -1

    for label in range(1, segment_num):
        (h_point, w_point) = np.where(segment_label == label)
        h1 = np.where(h_point == 0)[0].shape[0]
        h2 = np.where(h_point == H-1)[0].shape[0]
        w1 = np.where(w_point == 0)[0].shape[0]
        w2 = np.where(w_point == W-1)[0].shape[0]

        edge_num = h1 + h2 + w1 + w2
        if (edge_num >= background_edge_num):
            background_edge_num = edge_num
            background_label = label
    
    i = np.ones((H, W), dtype=np.uint8) * 255
    i[segment_label == background_label] = 0

    return i

def select_sign(img):
    i = pre_proc(img)
    i = segmentBackground(i)
    return i

def eval(n=1):
    import matplotlib.pyplot as plt
    i = cv.imread(f"datasets/train_label/{n}.bmp", cv.IMREAD_GRAYSCALE)
    i = select_sign(i)
    plt.imshow(i)
    plt.show()