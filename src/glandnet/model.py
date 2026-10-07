"""4-level U-Net with a 2-channel output (gland interior + gland border)."""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet4L(nn.Module):
    """Input [B,3,H,W] (H, W divisible by 16) -> logits [B,out_ch,H,W]."""

    def __init__(self, in_ch: int = 3, out_ch: int = 2):
        super().__init__()
        self.down1, self.down2 = DoubleConv(in_ch, 64), DoubleConv(64, 128)
        self.down3, self.down4 = DoubleConv(128, 256), DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4, self.conv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2), DoubleConv(1024, 512)
        self.up3, self.conv3 = nn.ConvTranspose2d(512, 256, 2, stride=2), DoubleConv(512, 256)
        self.up2, self.conv2 = nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128)
        self.up1, self.conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))
        bn = self.bottleneck(self.pool(c4))
        c4 = self.conv4(torch.cat([self.up4(bn), c4], dim=1))
        c3 = self.conv3(torch.cat([self.up3(c4), c3], dim=1))
        c2 = self.conv2(torch.cat([self.up2(c3), c2], dim=1))
        c1 = self.conv1(torch.cat([self.up1(c2), c1], dim=1))
        return self.final(c1)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
