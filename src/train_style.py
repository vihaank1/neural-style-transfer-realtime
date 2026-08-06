import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T

from autoencoder import FeatureEncoder
from transformer_net import TransformerNet
from data import ImageFolderFlat, load_file_list, train_transform, IMG_SIZE
from losses import gram_matrix, total_variation

def load_style_image(path, device):
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    return tf(img).unsqueeze(0).to(device)

def main(list_path, style_path, encoder_ckpt, out_ckpt, epochs=6, batch_size=8,
         lr=1e-3, content_weight=1.0, style_weight=5e4, tv_weight=1e-4,
         log_path=None, limit=None, init_ckpt=None, epoch_offset=0):
    torch.manual_seed(0)
    device = torch.device("cpu")

    encoder = FeatureEncoder().to(device)
    raw_sd = torch.load(encoder_ckpt, map_location=device)
    enc_sd = {k[len("encoder."):]: v for k, v in raw_sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd if enc_sd else raw_sd)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    net = TransformerNet().to(device)
    if init_ckpt and os.path.exists(init_ckpt):
        net.load_state_dict(torch.load(init_ckpt, map_location=device))
        print('resumed weights from', init_ckpt)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    style_img = load_style_image(style_path, device)
    with torch.no_grad():
        style_feats = encoder(style_img, return_features=True)
        style_grams = {k: gram_matrix(v) for k, v in style_feats.items()}

    files = load_file_list(list_path)
    if limit:
        files = files[:limit]
    ds = ImageFolderFlat(files, transform=train_transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        c_loss_sum = s_loss_sum = tv_loss_sum = 0.0
        n_batches = 0
        for batch in dl:
            batch = batch.to(device)
            opt.zero_grad()

            stylized = net(batch)

            with torch.no_grad():
                content_feats = encoder(batch, return_features=True)
            stylized_feats = encoder(stylized, return_features=True)

            content_loss = nn.functional.mse_loss(stylized_feats["f2"], content_feats["f2"])

            style_loss = 0.0
            b = batch.shape[0]
            for k in stylized_feats:
                g_stylized = gram_matrix(stylized_feats[k])
                g_style = style_grams[k].expand(b, -1, -1)
                style_loss = style_loss + nn.functional.mse_loss(g_stylized, g_style)

            tv_loss = total_variation(stylized)

            loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
            loss.backward()
            opt.step()

            c_loss_sum += content_loss.item()
            s_loss_sum += style_loss.item()
            tv_loss_sum += tv_loss.item()
            n_batches += 1

        elapsed = time.time() - t0
        avg_c, avg_s, avg_tv = c_loss_sum / n_batches, s_loss_sum / n_batches, tv_loss_sum / n_batches
        global_epoch = epoch_offset + epoch + 1
        print(f"[style] epoch {global_epoch} content={avg_c:.5f} style={avg_s:.6f} tv={avg_tv:.5f} elapsed={elapsed:.1f}s", flush=True)
        history.append({"epoch": global_epoch, "content_loss": avg_c, "style_loss": avg_s,
                         "tv_loss": avg_tv, "elapsed_sec": elapsed})

    torch.save(net.state_dict(), out_ckpt)
    if log_path:
        prior = []
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    prior = json.load(f)
            except Exception:
                prior = []
        with open(log_path, "w") as f:
            json.dump(prior + history, f, indent=2)
    print("saved checkpoint to", out_ckpt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", default="/tmp/style_transfer/selected_list.txt")
    p.add_argument("--style", default="/tmp/style_transfer/data/style/mosaic.jpg")
    p.add_argument("--encoder_ckpt", default="/tmp/style_transfer/checkpoints/autoencoder.pt")
    p.add_argument("--out", default="/tmp/style_transfer/checkpoints/style_mosaic.pt")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--style_weight", type=float, default=5e4)
    p.add_argument("--log", default="/tmp/style_transfer/checkpoints/style_mosaic_log.json")
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--epoch_offset", type=int, default=0)
    args = p.parse_args()
    main(args.list, args.style, args.encoder_ckpt, args.out, epochs=args.epochs,
         batch_size=args.batch_size, limit=args.limit, style_weight=args.style_weight, log_path=args.log,
         init_ckpt=args.init_ckpt, epoch_offset=args.epoch_offset)
