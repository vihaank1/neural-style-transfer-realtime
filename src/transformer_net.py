import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.pad(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

class UpsampleConvLayer(nn.Module):
    """Nearest-neighbor upsample + conv, avoids checkerboard artifacts from
    transposed convolutions."""
    def __init__(self, in_c, out_c, kernel_size, stride, upsample=2):
        super().__init__()
        self.upsample = upsample
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.conv(self.pad(x))

class TransformerNet(nn.Module):
    """Feed-forward image transformation network (Johnson et al., 2016 style).
    Trained from scratch here; a single forward pass stylizes an image,
    which is what makes this deployable in a real-time video pipeline."""
    def __init__(self, n_res_blocks=5):
        super().__init__()
        # downsampling
        self.conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(64, affine=True)
        # residual blocks
        self.res = nn.Sequential(*[ResidualBlock(64) for _ in range(n_res_blocks)])
        # upsampling
        self.deconv1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(16, affine=True)
        self.deconv3 = ConvLayer(16, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return torch.sigmoid(y)
