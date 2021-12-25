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

def pre_proc(img, th=10):
    i = cv.GaussianBlur(img, (3, 3), 0)
    i = img
    p1 = i >= th
    p2 = i < th
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

def selectSign(segment_img, threshold=0.02):
    ''' Get connected components and select the component which size is larger than the threshold '''
    (H, W) = segment_img.shape
    S = H * W
    (segment_num, segment_labels, stats, centroids) = cv.connectedComponentsWithStats(segment_img, connectivity=4)
    (x, y, w, h) = (stats[:,0], stats[:,1], stats[:,2], stats[:,3]) 
    select_ptr = (w*h) / S >= threshold
    (x1, x2, y1, y2) = (x[select_ptr], x[select_ptr] + w[select_ptr], y[select_ptr], y[select_ptr] + h[select_ptr])
    return (x1[1:], x2[1:], y1[1:], y2[1:]) # Select [1:] to ignore background

def getCoordinate(img):
    i = pre_proc(img)
    i = segmentBackground(i)
    (x1, x2, y1, y2) = selectSign(i)
    return (x1, x2, y1, y2)
