from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Small, straightforward U-Net for 3-channel 256x256 images ---
# Keeps params ~1–2M so it trains fast on CPU if needed.

class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # use bilinear upsample + 1x1 to reduce checkerboard artifacts
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1x1 = nn.Conv2d(in_c // 2, in_c // 2, kernel_size=1)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed (in case of odd dims)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY or diffX:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        # concatenate along channels
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNNetSmall(nn.Module):
    """
    Small U-Net variant.
    in_ch=3 -> features [32, 64, 128, 256] -> out_ch (default 1)
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        f1, f2, f3, f4 = base, base*2, base*4, base*8

        self.inc = DoubleConv(in_ch, f1)
        self.down1 = Down(f1, f2)
        self.down2 = Down(f2, f3)
        self.down3 = Down(f3, f4)

        self.up1 = Up(f4 + f3, f3)
        self.up2 = Up(f3 + f2, f2)
        self.up3 = Up(f2 + f1, f1)

        self.outc = nn.Conv2d(f1, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)     # 1/1
        x2 = self.down1(x1)  # 1/2
        x3 = self.down2(x2)  # 1/4
        x4 = self.down3(x3)  # 1/8

        x = self.up1(x4, x3) # 1/4
        x = self.up2(x,  x2) # 1/2
        x = self.up3(x,  x1) # 1/1
        x = self.outc(x)
        return x

if __name__ == "__main__":
    # quick sanity check
    m = UNNetSmall()
    y = m(torch.randn(2, 3, 256, 256))
    print(y.shape)  # -> (2, 1, 256, 256)