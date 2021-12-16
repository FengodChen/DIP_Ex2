from torch.utils.data.dataloader import DataLoader
from data.Datasets import TrainDataset, TestDataset
from utils.Compare_Func import void_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
from utils.Net import *
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

torch.backends.cudnn.benchmark = True

def test_train():
    dataset_train = TrainDataset("datasets")
    dataloader_train = DataLoader(dataset_train, batch_size=32)

    net = Net()
    logger = Logger("save/net", net, load_newest=True)

    for (x, y) in dataloader_train:
        for k in range(14):
            plt.subplot(131)
            plt.imshow(x[k].permute(1, 2, 0).detach().numpy())

            plt.subplot(132)
            plt.imshow(y[k].detach().numpy())

            plt.subplot(133)
            plt.imshow(net(x)[k].permute(1, 2, 0).detach().numpy())

            plt.savefig(f"save/train/{k}.jpg")
        break

    

def test_test():
    dataset_test = TestDataset("datasets")
    dataloader_train = DataLoader(dataset_test, batch_size=10)

    net = Net()
    logger = Logger("save/net", net, load_newest=True)
    #logger = Logger("save", net, timestamp=20211215210509)
    for (x, y) in dataloader_train:
        for k in range(10):
            plt.subplot(121)
            plt.imshow(x[k].permute(1, 2, 0).detach().numpy())

            plt.subplot(122)
            plt.imshow(net(x)[k].permute(1, 2, 0).detach().numpy())

            plt.savefig(f"save/test/{k}.jpg")
        break

if __name__ == "__main__":
    test_train()
    test_test()