from torch.utils.data.dataloader import DataLoader
from data.Datasets import TrainDataset, TestDataset
from utils.Compare_Func import void_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
from img_proc.Segment_Net import *
from img_proc.Select_Algo import *
from img_proc.Classification_Algo import *

import torch
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

gpu = torch.device("cuda")
cpu = torch.device("cpu")
net = Net_With_Select().to(gpu)
logger = Logger("save/net_with_select", net, load_newest=True)

def draw(dataloader, batch_size, save_path):
    for (x, y) in dataloader:
        for k in range(batch_size):
            print(f"Drawing {save_path}/{k+1}.jpg")
            orgin_img = np.uint8(x[k].permute(1, 2, 0).detach().numpy() * 255)
            draw_img = orgin_img.copy()

            net_seg = net(x.to(gpu))[k].to(cpu).permute(1, 2, 0).detach().numpy()

            cv_array = np.uint8(net_seg * 255)
            (H, W, _) = cv_array.shape
            cv_array = cv_array.reshape(H, W)

            cv_array_pre = pre_proc(cv_array)
            cv_seg = segmentBackground(cv_array_pre)

            (x1, x2, y1, y2, label_list, orgin_seg, canny_img) = classification(orgin_img, cv_seg)
            for (x_start, x_end, y_start, y_end, label) in zip(x1, x2, y1, y2, label_list):
                if (label == 1):
                    draw_img = cv.rectangle(draw_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
                    draw_img = cv.putText(draw_img, "C", (x_end, y_end), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv.LINE_AA)
                elif (label == 2):
                    draw_img = cv.rectangle(draw_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
                    draw_img = cv.putText(draw_img, "R", (x_end, y_end), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv.LINE_AA)
                else:
                    draw_img = cv.rectangle(draw_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
                    draw_img = cv.putText(draw_img, "N", (x_end, y_end), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv.LINE_AA)
            
            plt.figure(figsize=(40, 10))

            plt.subplot(161)
            plt.imshow(orgin_img)

            plt.subplot(162)
            plt.imshow(cv_array, cmap='gray', vmin=0, vmax=255)

            plt.subplot(163)
            plt.imshow(cv_seg, cmap='gray', vmin=0, vmax=255)

            plt.subplot(164)
            plt.imshow(orgin_seg, cmap='gray', vmin=0, vmax=255)

            plt.subplot(165)
            plt.imshow(canny_img, cmap='gray', vmin=0, vmax=255)

            plt.subplot(166)
            plt.imshow(draw_img, cmap='gray', vmin=0, vmax=255)

            plt.savefig(f"{save_path}/{k+1}.jpg")
        break

def drawTrain():
    batch_size = 14
    dataset_train = TrainDataset("datasets")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    draw(dataloader_train, batch_size, "save/train")

def drawTest():
    batch_size = 10
    dataset_test = TestDataset("datasets")
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)
    draw(dataloader_test, batch_size, "save/test")

if __name__ == "__main__":
    drawTrain()
    drawTest()