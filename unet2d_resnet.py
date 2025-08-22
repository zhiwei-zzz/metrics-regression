import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic residual conv block
class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.skip(x)
        return F.relu(out + skip)

# U-Net with residual blocks
class ResUNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(64,128,256,512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_ch
        for f in features:
            self.encoders.append(ResidualBlock2D(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f

        self.bottleneck = ResidualBlock2D(features[-1], features[-1]*2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_features = list(reversed(features))
        prev_ch = features[-1]*2
        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock2D(prev_ch, f))
            prev_ch = f

        self.head = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:  # safe crop
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)

# Example usage
if __name__ == "__main__":
    model = ResUNet2D(in_ch=3, out_ch=2)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)   # -> (1, 2, 256, 256)
