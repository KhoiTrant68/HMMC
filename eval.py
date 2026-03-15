import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    print("Please install pytorch-msssim: pip install pytorch-msssim")
    sys.exit(1)

from model.hmmc import HMMC

warnings.filterwarnings("ignore")


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    ms_val = ms_ssim(a, b, data_range=1.0).item()
    return -10 * math.log10(max(1 - ms_val, 1e-10))


def compute_bpp_estimated(out_net, num_pixels):
    bpp = 0
    for likelihoods in out_net["likelihoods"].values():
        bpp += torch.log2(likelihoods).sum() / (-num_pixels)
    return bpp.item()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="HMMC Evaluation Script")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to dataset folder"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save outputs"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--real", action="store_true", help="Actual compression")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"

    if device == "cuda:0":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    data_path = Path(args.data)
    save_path = Path(args.save_path) if args.save_path else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".png", ".jpeg", ".bmp"}
    img_list = [f for f in data_path.iterdir() if f.suffix.lower() in extensions]
    img_list.sort()

    print(f"Loading model from {args.checkpoint}...")
    net = HMMC(N=192, M=320).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = {
        k.replace("module.", ""): v
        for k, v in checkpoint.get("state_dict", checkpoint).items()
    }
    net.load_state_dict(state_dict)
    net.eval()

    if args.real:
        print("Updating entropy bottlenecks (CDFs)...")
        net.update(force=True)

    metrics = {
        k: 0.0 for k in ("psnr", "ms_ssim", "bpp", "time_total", "time_enc", "time_dec")
    }
    count = 0
    to_tensor = transforms.ToTensor()

    print(f"Starting inference on {len(img_list)} images...")
    for img_file in img_list:
        img = Image.open(img_file).convert("RGB")
        x = to_tensor(img).unsqueeze(0).to(device)

        orig_shape = x.size()
        orig_h, orig_w = orig_shape[2], orig_shape[3]
        num_pixels = orig_shape[0] * orig_h * orig_w

        with torch.no_grad():
            if args.real:
                if args.cuda:
                    torch.cuda.synchronize()
                t_start = time.time()
                out_enc = net.compress(x)
                if args.cuda:
                    torch.cuda.synchronize()
                t_enc = time.time() - t_start

                if args.cuda:
                    torch.cuda.synchronize()
                t_start = time.time()
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                t_dec = time.time() - t_start

                # Safe crop applied just in case ConvTranspose2d output_padding creates a +1 spatial dimension mismatch
                x_hat = out_dec["x_hat"][:, :, :orig_h, :orig_w].clamp_(0, 1)

                y_bits = len(out_enc["strings"][0][0]) * 8.0
                z_bits = sum(len(s) for s in out_enc["strings"][1]) * 8.0
                current_bpp = (y_bits + z_bits) / num_pixels

                metrics["time_enc"] += t_enc
                metrics["time_dec"] += t_dec
                metrics["time_total"] += t_enc + t_dec
            else:
                if args.cuda:
                    torch.cuda.synchronize()
                t_start = time.time()
                out_net = net(x, training_mode="ste")
                if args.cuda:
                    torch.cuda.synchronize()
                t_total = time.time() - t_start

                # Safe crop applied just in case ConvTranspose2d output_padding creates a +1 spatial dimension mismatch
                x_hat = out_net["x_hat"][:, :, :orig_h, :orig_w].clamp_(0, 1)

                current_bpp = compute_bpp_estimated(out_net, num_pixels)
                metrics["time_total"] += t_total

        current_psnr = compute_psnr(x, x_hat)
        current_msssim = compute_msssim(x, x_hat)

        count += 1
        metrics["psnr"] += current_psnr
        metrics["ms_ssim"] += current_msssim
        metrics["bpp"] += current_bpp

        print(
            f"[{count}/{len(img_list)}] {img_file.name} | Bpp: {current_bpp:.3f} | PSNR: {current_psnr:.2f} | MS-SSIM: {current_msssim:.2f}"
        )

        if save_path:
            save_image(x_hat, save_path / f"recon_{img_file.name}")

    print("-" * 40)
    print(f"Results ({count} images):")
    print(f"Avg PSNR:      {metrics['psnr'] / count:.2f} dB")
    print(f"Avg MS-SSIM:   {metrics['ms_ssim'] / count:.4f}")
    print(f"Avg Bitrate:   {metrics['bpp'] / count:.3f} bpp")


if __name__ == "__main__":
    main(sys.argv[1:])
