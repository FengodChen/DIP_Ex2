import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, path) -> None:
        data_path = f"{path}/train/"
        label_path_ident = f"{path}/label_ident/"
        label_path_word = f"{path}/label_word/"
        file_name = os.listdir(data_path)

        x_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((512, 512))
        ])
        y_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512))
        ])
        self.x = [x_trans(Image.open(f"{data_path}/{f_name}")) for f_name in file_name]
        self.y1 = [y_trans(Image.open(f"{label_path_ident}/{f_name}")) for f_name in file_name]
        self.y2 = [y_trans(Image.open(f"{label_path_word}/{f_name}")) for f_name in file_name]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y1[index], self.y2[index])

class TestDataset(Dataset):
    def __init__(self, path) -> None:
        data_path = f"{path}/test/"
        file_name = os.listdir(data_path)

        x_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((512, 512))
        ])
        self.x = [x_trans(Image.open(f"{data_path}/{f_name}")) for f_name in file_name]
        self.y = [0 for f_name in file_name]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])