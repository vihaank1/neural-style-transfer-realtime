import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image, ImageDraw, ImageFont
from transformer_net import TransformerNet
from infer import stylize_image

device = torch.device("cpu")
model = TransformerNet().to(device)
model.load_state_dict(torch.load("/tmp/style_transfer/checkpoints/style_mosaic.pt", map_location=device))
model.eval()

with open("/tmp/style_transfer/holdout_final.txt") as f:
    paths = [l.strip() for l in f if l.strip()]

pairs = [stylize_image(model, p, device) for p in paths]
w, h = pairs[0][0].size
pad = 6
label_h = 20
cols = 2
rows = len(pairs)
grid = Image.new("RGB", (w * cols + pad * (cols + 1), (h + label_h) * rows + pad * (rows + 1)), "white")
draw = ImageDraw.Draw(grid)
for i, (a, b) in enumerate(pairs):
    y = pad + i * (h + label_h + pad)
    grid.paste(a, (pad, y + label_h))
    grid.paste(b, (pad * 2 + w, y + label_h))
    draw.text((pad, y), "content", fill="black")
    draw.text((pad * 2 + w, y), "stylized (ours)", fill="black")

grid.save("/tmp/style_transfer/samples/final_comparison.jpg", quality=92)
print("saved final_comparison.jpg")
