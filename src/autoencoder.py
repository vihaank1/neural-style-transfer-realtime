import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2, output_padding=1, act=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=stride,
                                        padding=1, output_padding=output_padding)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.norm(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x

class FeatureEncoder(nn.Module):
    """Self-supervised encoder: 128x128x3 -> 16x16x128 feature map.
    Trained via reconstruction (autoencoder); intermediate activations later
    serve as the perceptual feature space for style-transfer losses, in place
    of an externally pretrained network."""
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 32)     # 128 -> 64
        self.block2 = ConvBlock(32, 64)    # 64 -> 32
        self.block3 = ConvBlock(64, 128)   # 32 -> 16

    def forward(self, x, return_features=False):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        if return_features:
            return {"f1": f1, "f2": f2, "f3": f3}
        return f3

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DeconvBlock(128, 64)               # 16 -> 32
        self.d2 = DeconvBlock(64, 32)                # 32 -> 64
        self.d3 = DeconvBlock(32, 3, act=False)       # 64 -> 128
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.out_act(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
