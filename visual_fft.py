import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.hmmc import HMMC


# ==========================================
# Helper: Compute 2D FFT for Visualization
# ==========================================
def compute_fft(feature_map):
    """
    Computes the log-magnitude 2D FFT of a feature map.
    feature_map: (H, W) numpy array
    """
    f = np.fft.fft2(feature_map)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    return magnitude_spectrum


# ==========================================
# Main Analysis Logic
# ==========================================
def analyze_frequencies(model, img_path, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # 1. Setup Hooks to capture Wavelet Outputs
    # We want to catch the output of the 'dwt' layer in the first spectral block
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            # output of dwt is a tuple: (ll, hf)
            activations[name] = output

        return hook

    # Hook the DWT of the first standard slice (usually contains the most structural info)
    target_layer = model.dt_cross_attention[0].dwt
    target_layer.register_forward_hook(get_activation("dwt_output"))

    # 2. Prepare Image
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    # Pad image (standard procedure for this architecture)
    h, w = x.shape[2], x.shape[3]
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64
    x_pad = F.pad(x, (0, pad_w, 0, pad_h))

    # 3. Forward Pass
    print(f"Running inference on {img_path}...")
    with torch.no_grad():
        model(x_pad)

    # 4. Retrieve Data
    ll, hf = activations["dwt_output"]

    # ll shape: [B, C, H/2, W/2]
    # hf shape: [B, 3*C, H/2, W/2] (Concatenated LH, HL, HH)

    # Collapse channels to get a "Heatmap" of activity
    # We use mean across channels to represent average energy
    ll_map = ll[0].mean(dim=0).cpu().numpy()

    # For HF, we average the 3 component sets (Vertical, Horizontal, Diagonal)
    # The channels are usually interleaved or concatenated.
    # Since hf is 3*C, we just mean across all to see general high-freq activity.
    hf_map = hf[0].mean(dim=0).cpu().numpy()

    # 5. Compute FFT (Frequency Domain)
    ll_fft = compute_fft(ll_map)
    hf_fft = compute_fft(hf_map)

    # ==========================================
    # Visualization
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original Image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Input")
    axes[0, 0].axis("off")

    # --- Row 1: Spatial Domain ---
    # Low Frequency Spatial
    im1 = axes[0, 1].imshow(ll_map, cmap="viridis")
    axes[0, 1].set_title("Low-Freq Features (Spatial)\n(Downsampled, Smooth)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # High Frequency Spatial
    im2 = axes[0, 2].imshow(hf_map, cmap="inferno")
    axes[0, 2].set_title("High-Freq Features (Spatial)\n(Edges, Textures)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --- Row 2: Frequency Domain (FFT) ---
    axes[1, 0].axis("off")  # Spacer
    axes[1, 0].text(
        0.5,
        0.5,
        "Frequency Domain Analysis\n(Center = DC/Low Freq)",
        ha="center",
        va="center",
        fontsize=12,
    )

    # Low Frequency FFT
    im3 = axes[1, 1].imshow(ll_fft, cmap="magma")
    axes[1, 1].set_title("Low-Freq Spectrum (FFT)\n(Energy concentrated in center)")
    axes[1, 1].axis("off")

    # High Frequency FFT
    im4 = axes[1, 2].imshow(hf_fft, cmap="magma")
    axes[1, 2].set_title("High-Freq Spectrum (FFT)\n(Energy spread to corners/edges)")
    axes[1, 2].axis("off")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "spectral_analysis.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Spectral analysis saved to {save_path}")

    # ==========================================
    # Quantitative Stats (For Paper Text)
    # ==========================================
    ll_energy = (ll**2).sum().item()
    hf_energy = (hf**2).sum().item()
    total_energy = ll_energy + hf_energy

    print("\n=== Quantitative Energy Analysis ===")
    print(f"Low-Frequency Energy:  {ll_energy:.2e} ({ll_energy/total_energy:.2%})")
    print(f"High-Frequency Energy: {hf_energy:.2e} ({hf_energy/total_energy:.2%})")
    print(
        "Interpretation: The MoE module only needs to process the High-Frequency percentage,"
    )
    print("saving compute on the majority Low-Frequency content.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to a test image")
    parser.add_argument("--save_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    # Load Model
    model = HMMC(N=192, M=320)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    analyze_frequencies(model, args.image, args.save_dir)
