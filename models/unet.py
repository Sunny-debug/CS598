# unchanged — pasted here for completeness if needed
import torch
import torch.nn as nn

def _conv_bn_relu(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(_conv_bn_relu(in_ch, base), _conv_bn_relu(base, base))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(_conv_bn_relu(base, base*2), _conv_bn_relu(base*2, base*2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(_conv_bn_relu(base*2, base*4), _conv_bn_relu(base*4, base*4))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(_conv_bn_relu(base*4, base*8), _conv_bn_relu(base*8, base*8))
        self.pool4 = nn.MaxPool2d(2)

        self.bott = nn.Sequential(_conv_bn_relu(base*8, base*16), _conv_bn_relu(base*16, base*16))

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = nn.Sequential(_conv_bn_relu(base*16, base*8), _conv_bn_relu(base*8, base*8))
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = nn.Sequential(_conv_bn_relu(base*8, base*4), _conv_bn_relu(base*4, base*4))
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = nn.Sequential(_conv_bn_relu(base*4, base*2), _conv_bn_relu(base*2, base*2))
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = nn.Sequential(_conv_bn_relu(base*2, base), _conv_bn_relu(base, base))
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bott(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], 1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        logits = self.outc(d1)
        return logits
