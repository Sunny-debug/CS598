import argparse, torch
from pathlib import Path
from models import UNNetSmall

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--size", type=int, default=256)
p.add_argument("--out", default=None)
a = p.parse_args()

sd = torch.load(a.ckpt, map_location="cpu")
model = UNNetSmall(in_ch=3, base=32)
model.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
model.eval()

example = torch.randn(1,3,a.size,a.size)
ts = torch.jit.trace(model, example)
out = a.out or (Path(a.ckpt).with_suffix(".ts.pt"))
ts.save(str(out))
print("Saved TorchScript to", out)