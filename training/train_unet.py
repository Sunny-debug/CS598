import argparse, os, torch
from torch.utils.data import DataLoader, random_split
from models.unet import UNNetSmall as _  # ensure import path correct if your package layout differs
from models.unet import UNetSmall
from training.datasets import InpaintSynthDataset
from training.losses import bce_dice_loss
from tqdm import tqdm

def train(args):
    ds = InpaintSynthDataset(args.src_dir, size=args.size)
    n_val = max(50, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall(in_ch=3, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.out_dir, exist_ok=True)
    best = 1e9

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        tr_loss = 0.0
        for img, mask in pbar:
            img = img.to(device)
            mask = mask.to(device)
            logits = model(img)
            loss = bce_dice_loss(logits, mask, bce_w=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * img.size(0)
            pbar.set_postfix(loss=loss.item())

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, mask in vl:
                img = img.to(device); mask = mask.to(device)
                logits = model(img)
                val_loss += bce_dice_loss(logits, mask).item() * img.size(0)
        val_loss /= len(val_ds)
        print(f"val_loss={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            ckpt = os.path.join(args.out_dir, "unet_best.pt")
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "args": vars(args)}, ckpt)
            print(f"Saved {ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="folder of clean images for synthetic edits")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()
    train(args)