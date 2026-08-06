import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image
import torchvision.transforms as T
from transformer_net import TransformerNet
from data import IMG_SIZE

def stylize_image(model, img_path, device):
    tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
    out = y.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray((out * 255).astype("uint8"))
    in_img = Image.fromarray((x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
    return in_img, out_img

def make_grid(pairs, out_path):
    n = len(pairs)
    w, h = pairs[0][0].size
    grid = Image.new("RGB", (w * 2, h * n))
    for i, (a, b) in enumerate(pairs):
        grid.paste(a, (0, i * h))
        grid.paste(b, (w, i * h))
    grid.save(out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/tmp/style_transfer/checkpoints/style_mosaic.pt")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", default="/tmp/style_transfer/samples/preview.jpg")
    args = p.parse_args()

    device = torch.device("cpu")
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    pairs = [stylize_image(model, path, device) for path in args.images]
    make_grid(pairs, args.out)
    print("saved", args.out)
