import cv2 as cv
import numpy as np

from img_proc.Select_Algo import pre_proc, segmentBackground, selectSign

def getImgFromFile(label):
    bgr_img = cv.imread(f"datasets/train/{label}.bmp")
    seg_img = cv.imread(f"datasets/train_label/{label}.bmp", cv.IMREAD_GRAYSCALE)
    return (bgr_img, seg_img)

def hasRectange(img, threshold=0.6):
    if (img is None):
        return False
    (H, W) = img.shape
    L = max(H, W)
    img = cv.resize(img, (L, L))
    L_MIN = int(L * threshold)
    lines = cv.HoughLines(img, 1, np.pi / 180, 50, None, L_MIN, 10)
    if (lines is None):
        return False
    elif (len(lines) >= 4):
        return True
    else:
        return False

def hasCircle(img, threshold=0.8):
    if (img is None):
        return False
    (H, W) = img.shape
    L = max(H, W)
    img = cv.resize(img, (L, L))
    L_MIN = int(L * threshold)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=L_MIN, maxRadius=L)
    if (circles is None):
        return False
    else:
        return True

def classification(bgr_img, seg_img):
    OTHER = 0
    CIRCLE = 1
    RECTANGE = 2
    label_list = []
    seg_img = pre_proc(seg_img)
    seg_img = segmentBackground(seg_img)
    (x1, x2, y1, y2) = selectSign(seg_img)
    seg_img[seg_img>=1] = 1
    for (w1, w2, h1, h2) in zip(x1, x2, y1, y2):
        gaus_seg_img = cv.GaussianBlur(bgr_img[h1:h2, w1:w2, :], (3, 3), 0)
        gray_seg_img = cv.cvtColor(gaus_seg_img, cv.COLOR_BGR2GRAY)
        gray_seg_img = np.uint8(gray_seg_img * seg_img[h1:h2, w1:w2])
        #gray_seg_img = np.uint8(seg_img[h1:h2, w1:w2] * 255)
        canny_seg_img = cv.Canny(gray_seg_img, 50, 150)
        if hasCircle(canny_seg_img):
            label_list.append(CIRCLE)
        elif hasRectange(canny_seg_img):
            label_list.append(RECTANGE)
        else:
            label_list.append(OTHER)

    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    gray_img = np.uint8(gray_img * seg_img)
    canny_img = cv.Canny(gray_img, 50, 150)
    return (x1, x2, y1, y2, label_list, gray_img, canny_img)
