from numpy.core.fromnumeric import argmax
from torch.utils.data.dataloader import DataLoader
import torchvision
from data.Datasets import ShapeDataset, TrainDataset, TestDataset
from img_proc.Classification_Algo import ShapeNet
from utils.Compare_Func import void_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
from img_proc.Segment_Net import *
import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", help='eval')
opt = parser.parse_args()

torch.backends.cudnn.benchmark = True

seed_everything(21120009)

gpu = torch.device("cuda")
cpu = torch.device("cpu")
net = ShapeNet(class_num=3).to(gpu)
logger = Logger("save/net_shape", net, load_newest=True)

def train():
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    dataset_train = ShapeDataset("datasets/shape")
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    trainer = Trainer(net, loss, opt, gpu, logger, 100)

    trainer.train(dataloader_train, dataloader_train, compare_func, 100, 1000)

def test():
    dataset_train = ShapeDataset("datasets/shape")
    dataloader_train = DataLoader(dataset_train, batch_size=32)
    kk = 0
    for (x, y) in dataloader_train:
        for k in range(y.shape[0]):
            kk += 1
            plt.subplot(111)
            plt.imshow(x[k].detach().numpy().reshape(64, 64))
            y_dt = int(torch.argmax(net(x.to(gpu))[k]).to(cpu).detach().numpy())

            plt.savefig(f"save/shape/{kk}-class-{y_dt}.jpg")


if __name__ == "__main__":
    if opt.eval:
        test()
    else:
        train()