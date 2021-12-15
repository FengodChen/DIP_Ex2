from torch.utils.data.dataloader import DataLoader
from data.Datasets import TrainDataset
from utils.Compare_Func import void_compare_func as compare_func
from utils.Seed import seed_everything
from data.Trainer import Trainer
from data.Logger import Logger
from utils.Net import *
import torch

torch.backends.cudnn.benchmark = True

seed_everything(21120009)

dev = torch.device("cuda")
dataset_train = TrainDataset("datasets")
dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)

net = Net().to(dev)
logger = Logger("save/net", net, load_newest=True)
loss = DiceLoss()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
trainer = Trainer(net, loss, opt, dev, logger, 100)

trainer.train(dataloader_train, dataloader_train, compare_func, 100, 10000)