import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        x = self.net1(x)
        return x 

class DiceLoss(nn.Module):
    def __init__(self, alpha = 0.3):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
