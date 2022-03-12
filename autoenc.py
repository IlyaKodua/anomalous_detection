import torch
import torch.nn as nn
import torch.nn.functional as F



class AE(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1))
        self.l2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1))
        self.l3 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1))
        self.l4 = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1))
        self.l5 = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1))
        self.l6 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1))
        self.l7 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1))
        self.l8 = nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh())

    def forward(self,x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)
        x6 = self.l6(x5)
        x7 = self.l7(x6)
        output = self.l8(x7)
        return x1, x2, x3, x5, x6, x7, output







class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)

        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = 1
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 32)
        self.down2 = Down(32, 32)
        self.down3 = Down(32, 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 32)
        self.up1 = Up(32, 32, bilinear)
        self.up2 = Up(32, 32, bilinear)
        self.up3 = Up(32, 32, bilinear)
        self.up4 = Up(32, 32, bilinear)
        self.outc = OutConv(32, 1)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.outc(x)