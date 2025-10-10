import argparse, torch
from models.unet import UNetSmall

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="unet.onnx")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    model = UNetSmall().eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    x = torch.randn(1,3,args.size,args.size)
    torch.onnx.export(model, x, args.out, input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"image":{0:"N"}, "logits":{0:"N"}})
    print(f"Exported -> {args.out}")