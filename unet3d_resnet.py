import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual conv block (3D)
class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.skip  = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.skip(x)
        return F.relu(out + skip)

# 3D ResUNet
class ResUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(32,64,128,256)):
        super().__init__()
        self.encoders, self.pools = nn.ModuleList(), nn.ModuleList()
        prev_ch = in_ch
        for f in features:
            self.encoders.append(ResidualBlock3D(prev_ch, f))
            self.pools.append(nn.MaxPool3d(2))
            prev_ch = f

        self.bottleneck = ResidualBlock3D(features[-1], features[-1]*2)

        self.upconvs, self.decoders = nn.ModuleList(), nn.ModuleList()
        rev_features = list(reversed(features))
        prev_ch = features[-1]*2
        for f in rev_features:
            self.upconvs.append(nn.ConvTranspose3d(prev_ch, f, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock3D(prev_ch, f))
            prev_ch = f

        self.head = nn.Conv3d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)

# Example usage
if __name__ == "__main__":
    model = ResUNet3D(in_ch=1, out_ch=3).cuda()
    x = torch.randn(1, 1, 64, 64, 64).cuda()  # (B, C, D, H, W)
    y = model(x)
    print(y.shape)   # -> (1, 3, 64, 64, 64)
