from torch.utils.data.dataloader import DataLoader
from data.Datasets import TrainDataset, TestDataset
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
net = Net_With_Select().to(gpu)
logger = Logger("save/net_with_select", net, load_newest=True)

def train():
    loss = DiceLoss()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    dataset_train = TrainDataset("datasets")
    dataloader_train = DataLoader(dataset_train, batch_size=14, shuffle=True)
    trainer = Trainer(net, loss, opt, gpu, logger, 100)

    trainer.train(dataloader_train, dataloader_train, compare_func, 100, 1000)

def test():
    dataset_train = TrainDataset("datasets")
    dataloader_train = DataLoader(dataset_train, batch_size=32)
    for (x, y) in dataloader_train:
        for k in range(14):
            plt.subplot(131)
            plt.imshow(x[k].permute(1, 2, 0).detach().numpy())

            plt.subplot(132)
            plt.imshow(y[k].detach().numpy())

            plt.subplot(133)
            plt.imshow(net(x.to(gpu))[k].to(cpu).permute(1, 2, 0).detach().numpy())

            plt.savefig(f"save/train/{k}.jpg")
        break

    dataset_test = TestDataset("datasets")
    dataloader_train = DataLoader(dataset_test, batch_size=10)
    for (x, y) in dataloader_train:
        for k in range(10):
            plt.subplot(121)
            plt.imshow(x[k].permute(1, 2, 0).detach().numpy())

            plt.subplot(122)
            plt.imshow(net(x.to(gpu))[k].to(cpu).permute(1, 2, 0).detach().numpy())

            plt.savefig(f"save/test/{k}.jpg")
        break

if __name__ == "__main__":
    if opt.eval:
        test()
    else:
        train()