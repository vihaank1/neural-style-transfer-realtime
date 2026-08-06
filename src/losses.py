import torch

def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)

def total_variation(img):
    dh = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dw = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dh + dw
