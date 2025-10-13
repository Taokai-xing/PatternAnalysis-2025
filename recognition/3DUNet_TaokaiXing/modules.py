"""
modules.py

Improved 3D UNet for Prostate Segmentation (COMP3710)
-----------------------------------------------------
Author: Taokai Xing

Based on:
- Cicek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (MICCAI 2016)
- Isensee et al., "nnU-Net: Self-adapting Framework for U-Net-based Medical Image Segmentation" (BRATS 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 🔹 Residual Convolutional Block with Instance Normalization
# ---------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.in1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels)

        # 1x1 conv to match dimensions if needed
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride) \
            if in_channels != out_channels else nn.Identity()

        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += residual
        return self.relu(out)


# ---------------------------------------------------------
# 🔹 Encoder Block: Residual Block + Downsampling
# ---------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ResidualConvBlock(in_channels, out_channels)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.block(x)
        skip = out
        out = self.down(out)
        return out, skip


# ---------------------------------------------------------
# 🔹 Decoder Block: Upsampling + Residual Block
# ---------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if necessary (for odd-sized volumes)
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


# ---------------------------------------------------------
# 🔹 Improved UNet3D
# ---------------------------------------------------------
class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, base_filters=32, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = ResidualConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters)

        # Output layers (for deep supervision)
        self.out1 = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        self.out2 = nn.Conv3d(base_filters * 2, out_channels, kernel_size=1)
        self.out3 = nn.Conv3d(base_filters * 4, out_channels, kernel_size=1)
        self.out4 = nn.Conv3d(base_filters * 8, out_channels, kernel_size=1)

        # Weight initialization
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1, s1 = self.enc1(x)
        e2, s2 = self.enc2(e1)
        e3, s3 = self.enc3(e2)
        e4, s4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        if self.deep_supervision:
            out1 = self.out1(d1)
            out2 = F.interpolate(self.out2(d2), size=out1.shape[2:], mode="trilinear", align_corners=False)
            out3 = F.interpolate(self.out3(d3), size=out1.shape[2:], mode="trilinear", align_corners=False)
            out4 = F.interpolate(self.out4(d4), size=out1.shape[2:], mode="trilinear", align_corners=False)
            return (out1 + out2 + out3 + out4) / 4
        else:
            return self.out1(d1)


# ---------------------------------------------------------
# 🔍 Quick Test
# ---------------------------------------------------------
if __name__ == "__main__":
    model = ImprovedUNet3D(in_channels=1, out_channels=6, deep_supervision=True)
    x = torch.randn(1, 1, 128, 128, 64)  # (Batch, Channels, H, W, D)
    y = model(x)
    print("Output shape:", y.shape)  # Expect (1, 6, H, W, D)
