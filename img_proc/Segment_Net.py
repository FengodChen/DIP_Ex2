import torch.nn as nn
import torch.nn.functional as F
import torch


class Net_Without_Select(nn.Module):
    def __init__(self, c_split=12, i_shape=(512, 512)):
        super().__init__()

        self.c_split = c_split
        (self.h, self.w) = (h, w) = i_shape

        # Split RGB Color
        self.split_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c_split, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=c_split, out_channels=c_split, kernel_size=1, stride=1, padding=0),
            nn.ReLU6(),
        )

        # Select Color Which May Influence Predict
        self.select_net = nn.Sequential(
            nn.Conv2d(in_channels=2*c_split, out_channels=4*c_split, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=4*c_split, out_channels=2*c_split, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*c_split, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.split_net(x) / 6.0
        x = torch.cat((x, x), dim=1)
        x = self.select_net(x)
        return x 
class Net_With_Select(nn.Module):
    def __init__(self, c_split=12, i_shape=(512, 512)):
        super().__init__()

        self.c_split = c_split
        (self.h, self.w) = (h, w) = i_shape

        # Split RGB Color
        self.split_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c_split, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=c_split, out_channels=c_split, kernel_size=1, stride=1, padding=0),
            nn.ReLU6(),
        )

        # Select Color Which May Influence Predict
        self.select_net = nn.Sequential(
            nn.Conv2d(in_channels=2*c_split, out_channels=4*c_split, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=4*c_split, out_channels=2*c_split, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*c_split, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.split_net(x) / 6.0

        space_info = x.sum(dim=(2, 3)) / (self.h * self.w) # shape = (batch size, channel)
        space_info = space_info.reshape(-1, self.c_split, 1, 1).expand(-1, self.c_split, self.h, self.w) # shape = (batch size, channel, h, w)

        x = torch.cat((x, space_info), dim=1)
        x = self.select_net(x)
        return x 

class Net_With_Select_Lite(nn.Module):
    def __init__(self, c_split=12, i_shape=(512, 512)):
        super().__init__()

        self.c_split = c_split
        (self.h, self.w) = (h, w) = i_shape

        # Split RGB Color
        self.split_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c_split, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=c_split, out_channels=c_split, kernel_size=1, stride=1, padding=0),
            nn.ReLU6(),
        )

        # Select Color Which May Influence Predict
        self.select_net = nn.Sequential(
            nn.Conv2d(in_channels=2*c_split, out_channels=c_split, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_split, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.split_net(x) / 6.0

        space_info = x.sum(dim=(2, 3)) / (self.h * self.w) # shape = (batch size, channel)
        space_info = space_info.reshape(-1, self.c_split, 1, 1).expand(-1, self.c_split, self.h, self.w) # shape = (batch size, channel, h, w)

        x = torch.cat((x, space_info), dim=1)
        x = self.select_net(x)
        return x 

class Net_Without_Select_Lite(nn.Module):
    def __init__(self, c_split=12, i_shape=(512, 512)):
        super().__init__()

        self.c_split = c_split
        (self.h, self.w) = (h, w) = i_shape

        # Split RGB Color
        self.split_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c_split, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=c_split, out_channels=c_split, kernel_size=1, stride=1, padding=0),
            nn.ReLU6(),
        )

        # Select Color Which May Influence Predict
        self.select_net = nn.Sequential(
            nn.Conv2d(in_channels=2*c_split, out_channels=c_split, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_split, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.split_net(x) / 6.0

        x = torch.cat((x, x), dim=1)
        x = self.select_net(x)
        return x 

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=8, stride=8, padding=0),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=8, stride=8),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
        )
        self.end = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            #nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        t = self.down(x)
        t = self.neck(t)
        t = self.up(t)
        x = torch.cat((t, x), dim=1)
        x = self.end(x)
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
        
        return abs(1 - dice)
