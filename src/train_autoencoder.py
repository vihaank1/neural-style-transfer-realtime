import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from data import ImageFolderFlat, load_file_list

def main(list_path, out_ckpt, epochs=8, batch_size=16, lr=1e-3, log_path=None, limit=None):
    torch.manual_seed(0)
    device = torch.device("cpu")
    files = load_file_list(list_path)
    if limit:
        files = files[:limit]
    ds = ImageFolderFlat(files)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = Autoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dl:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = crit(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"[autoencoder] epoch {epoch+1}/{epochs}  loss={avg_loss:.5f}  elapsed={elapsed:.1f}s", flush=True)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "elapsed_sec": elapsed})

    torch.save(model.state_dict(), out_ckpt)
    if log_path:
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
    print("saved checkpoint to", out_ckpt)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--list", default="/tmp/style_transfer/selected_list.txt")
    p.add_argument("--out", default="/tmp/style_transfer/checkpoints/autoencoder.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log", default="/tmp/style_transfer/checkpoints/autoencoder_log.json")
    args = p.parse_args()
    main(args.list, args.out, epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, log_path=args.log)
