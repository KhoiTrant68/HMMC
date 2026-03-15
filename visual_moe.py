import argparse
import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- LOCAL MODULE IMPORTS ---
from model.hmmc import HMMC


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HMMC MoE Router Decisions")
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth.tar)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="vis_results", help="Directory to save results"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Opacity of the expert overlay (0.0 to 1.0)",
    )
    return parser.parse_args()


def load_image(path, device):
    """
    Loads an image and pads it to be divisible by 64.
    Returns: padded_tensor, (original_height, original_width)
    """
    img = Image.open(path).convert("RGB")
    original_h, original_w = img.height, img.width

    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Pad to multiple of 64 (Model requirement for 6 downsampling layers)
    h, w = x.shape[2:]
    p_h = (64 - (h % 64)) % 64
    p_w = (64 - (w % 64)) % 64

    # F.pad format: (left, right, top, bottom)
    if p_h > 0 or p_w > 0:
        x = F.pad(x, (0, p_w, 0, p_h), mode="reflect")

    return x, (original_h, original_w)


def clean_state_dict(state_dict):
    """Removes 'module.' prefix usually added by DataParallel/Accelerate."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict


def get_model_params_from_state_dict(state_dict):
    """
    Infers N and M parameters by inspecting weight shapes.
    """
    try:
        if "h_a.0.conv_down.weight" in state_dict:
            N = state_dict["h_a.0.conv_down.weight"].shape[0]
            M = state_dict["h_a.0.conv_down.weight"].shape[1]
            print(f"   [Auto-Detect] Hyperparameters found: N={N}, M={M}")
            return N, M
        elif "g_a.6.weight" in state_dict:
            M = state_dict["g_a.6.weight"].shape[0]
            print(f"   [Auto-Detect] M={M} found. Defaulting N=192.")
            return 192, M
    except Exception as e:
        print(f"   [Warning] Could not infer params: {e}")

    print("[Warning] Using default N=192, M=320.")
    return 192, 320


def get_expert_colormap(num_experts=4):
    """Defines colors for experts: Red, Green, Blue, Yellow."""
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]
    if num_experts > 4:
        extra_colors = ["#00FFFF", "#FF00FF", "#FFFFFF", "#000000"]
        colors.extend(extra_colors[: num_experts - 4])
    return mcolors.ListedColormap(colors[:num_experts])


def overlay_expert_map(img_tensor, expert_indices, alpha=0.4, num_experts=4):
    """
    Overlays expert choices onto the image.
    img_tensor:[1, 3, H, W] (The reconstructed image)
    expert_indices: [H_feat, W_feat] (The latent map)
    """
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    H, W, _ = img_np.shape

    # Resize expert map to Image Size using Nearest Neighbor
    expert_map_tensor = expert_indices.unsqueeze(0).unsqueeze(0).float()
    expert_map_up = F.interpolate(expert_map_tensor, size=(H, W), mode="nearest")
    expert_map_up = expert_map_up.squeeze().cpu().numpy()  # [H, W]

    cmap = get_expert_colormap(num_experts)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, num_experts + 0.5, 1), cmap.N)

    colored_map = cmap(norm(expert_map_up))
    overlay_rgb = colored_map[..., :3]

    # Blend: (1-alpha)*Original + alpha*Overlay
    blended = (1 - alpha) * img_np + alpha * overlay_rgb
    blended = np.clip(blended, 0, 1)

    return blended, expert_map_up


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("WARNING: CUDA not found. Mamba/Triton kernels usually fail on CPU.")
        device = "cpu"
    print(f"1. Device set to: {device}")

    print(f"2. Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)

    N, M = get_model_params_from_state_dict(state_dict)
    model = HMMC(N=N, M=M)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(
            f"   [Warning] Strict loading failed. Retrying with strict=False.\n   Error: {e}"
        )
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # Important: Update scale tables for entropy model
    model.update(force=True)

    x_padded, (orig_h, orig_w) = load_image(args.image, device)
    print(
        f"3. Image loaded. Original: {orig_h}x{orig_w}, Padded: {x_padded.shape[2]}x{x_padded.shape[3]}"
    )

    print("4. Running inference...")
    with torch.no_grad():
        # Use 'ste' (Straight-Through Estimator) to simulate quantization rounding
        out = model(x_padded, training_mode="ste")

        x_hat_padded = out["x_hat"]
        router_logits = out["router_logits"]

    if router_logits is None:
        print(
            "Error: No router logits returned. The model might not be using MoE layers."
        )
        return

    # Unpad / Crop Logic
    x_hat = x_hat_padded[:, :, :orig_h, :orig_w].clamp_(0, 1)
    x_orig = x_padded[:, :, :orig_h, :orig_w]

    # ACCURATE METRICS: Calculate specifically on unpadded dimensions to prevent BPP leakage
    mse_val = torch.mean((x_hat - x_orig) ** 2).item()
    psnr_val = -10 * math.log10(mse_val) if mse_val > 0 else 100.0

    num_pixels = orig_h * orig_w
    bpp_val = 0.0
    for likelihoods in out["likelihoods"].values():
        if isinstance(likelihoods, torch.Tensor):
            bpp_val += torch.log2(likelihoods.clamp(min=1e-9)).sum().item() / (
                -num_pixels
            )
        else:
            for l in likelihoods:
                bpp_val += torch.log2(l.clamp(min=1e-9)).sum().item() / (-num_pixels)

    print(f"5. Metrics Calculated -> BPP: {bpp_val:.4f}, PSNR: {psnr_val:.2f} dB")

    # Visualization Plotting
    num_layers = len(router_logits)
    cols = 4
    total_items = 3 + num_layers
    rows = math.ceil(total_items / cols)

    fig = plt.figure(figsize=(20, 5 * rows))

    # --- Plot 1: Original ---
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(x_orig.squeeze().permute(1, 2, 0).cpu().numpy())
    ax.set_title("Original Input")
    ax.axis("off")

    # --- Plot 2: Reconstruction ---
    ax = fig.add_subplot(rows, cols, 2)
    x_hat_np = x_hat.squeeze().permute(1, 2, 0).cpu().numpy()
    ax.imshow(x_hat_np)
    ax.set_title(f"Reconstruction\nBPP: {bpp_val:.3f} | PSNR: {psnr_val:.2f} dB")
    ax.axis("off")

    # --- Plot 3: Legend ---
    ax = fig.add_subplot(rows, cols, 3)
    cmap = get_expert_colormap(4)
    patches = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(4)]
    ax.legend(
        patches, [f"Expert {i}" for i in range(4)], loc="center", fontsize="large"
    )
    ax.axis("off")
    ax.set_title("Expert Color Map")

    # --- Plot MoE Layers ---
    layer_names = [f"Standard Slice {i}" for i in range(num_layers - 2)] + [
        "Anchor Slice",
        "Non-Anchor Slice",
    ]

    for i, layer_data in enumerate(router_logits):
        if layer_data is None:
            continue

        # indices shape:[Batch, H_latent, W_latent, TopK]
        logits, indices = layer_data

        # Get Top-1 Expert Choice map
        expert_map_latent = indices[0, :, :, 0]

        # Overlay on padded reconstruction first to ensure scaling alignment
        blended_padded, raw_map = overlay_expert_map(
            x_hat_padded.clamp(0, 1), expert_map_latent, alpha=args.alpha
        )

        # Crop the result to remove padding area
        blended_cropped = blended_padded[:orig_h, :orig_w, :]

        # Calculate Usage Stats on the whole latent map
        unique, counts = np.unique(raw_map, return_counts=True)
        total_px = raw_map.size
        stats_str = ", ".join(
            [f"E{int(u)}:{c/total_px:.0%}" for u, c in zip(unique, counts)]
        )

        ax = fig.add_subplot(rows, cols, i + 4)
        ax.imshow(blended_cropped)

        t_name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
        ax.set_title(f"{t_name}\nTop-1 Usage: {stats_str}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    save_filename = f"vis_{os.path.basename(args.image).split('.')[0]}.png"
    save_path = os.path.join(args.save_dir, save_filename)
    plt.savefig(save_path, dpi=150)
    print(f"6. Visualization saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
