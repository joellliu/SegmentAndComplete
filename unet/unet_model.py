""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base_filter=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_filter)
        self.down1 = Down(base_filter, base_filter*2)
        self.down2 = Down(base_filter*2, base_filter*2**2)
        self.down3 = Down(base_filter*2**2, base_filter*2**3)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filter*2**3, base_filter*2**4 // factor)
        self.up1 = Up(base_filter*2**4, base_filter*2**3 // factor, bilinear)
        self.up2 = Up(base_filter*2**3, base_filter*2**2 // factor, bilinear)
        self.up3 = Up(base_filter*2**2, base_filter*2 // factor, bilinear)
        self.up4 = Up(base_filter*2, base_filter, bilinear)
        self.outc = OutConv(base_filter, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
